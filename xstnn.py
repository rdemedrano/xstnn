import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from module import MLP
from utils import identity


class xSpatioTemporalNN(nn.Module):
    def __init__(self, relations, exogenous, nx, nt, np, nd, nz, mode=None, nhid=0, nlayers=1, dropout_f=0., dropout_d=0.,
                 activation='tanh', periode=1):
        super(xSpatioTemporalNN, self).__init__()
        assert (nhid > 0 and nlayers > 1) or (nhid == 0 and nlayers == 1)
        # attributes
        self.nt = nt
        self.nx = nx
        self.np = np
        self.nz = nz
        self.mode = mode
        self.exogenous = exogenous
        # kernel
        # Creo que la activación está bastante clara.
        self.activation = torch.tanh if activation == 'tanh' else identity if activation == 'identity' else None
        device = relations.device
        # A la hora de obtener las relaciones hay dos posibilidades: que estemos en modo de descubrirlas, o en normal-refinarlas
        # En este segundo caso simplemente se utiliza el relations conocido
        # Téngase en cuenta que torch.cat concatena vectores en cierta dimensión, torch.eye devuelve un tensor 2D diagonal de
        # 1s de número de filas nx, unsqeeeze(1) devuelve un tensor (no tengo muy claro que hace).
        # En definitiva, torch.eye(nx).to(device).unsqueeze(1) saca un tensor diagonal de dimensiones [nx, 1, nx], y el cat con
        # el relations genera un tensor raruno de [nx, 2, nx]. Por la pinta que tiene, es como que primero va una lista que marca
        # la posición espacial (tiene un 1 en la posición que estés mirando, es decir tiene un 1 en la posisición 4 si haces
        # relation[4] por ejemplo), y luego va otra lista que es la fila de relations original que corresponda. Entiendo que
        # necesariamente serán listas de filas siempre y que dependerá de la dimensionalidad, pero así se ve claro.
        # Si estás en discover generas algo similar pero lleno de unos la lista que da los valores (no las posiciones).
        if mode is None or mode == 'refine':
            self.relations = torch.cat((torch.eye(nx).to(device).unsqueeze(1), relations), 1)
        elif mode == 'discover':
            self.relations = torch.cat((torch.eye(nx).to(device).unsqueeze(1),
                                        torch.ones(nx, 1, nx).to(device)), 1)
        self.nr = self.relations.size(1)
        # modules
        # Como linear, defines la función drop que hará 0 elementos del tensor que pases como parámetros con probabilidad
        # dropout_f
        self.drop = nn.Dropout(dropout_f)
        # Se recuerda que las ns hacen referencia al tamaño de los espacios temporales, espaciales y latente.
        # Este último es por defecto 1.   
        # Me da que torch.Tensor inicializa un tensor de cero, más bien números muy grandes o muy pequeños. NO importa,
        # porque en _init_weights los inicializas entre -0.1 y +0.1, o lo que toque. Es decir, creo que son los "pesos" o 
        # o valores de z.
        # Parameter simplementes es una forma de definir un tensor con parámetros en pytorch, que dentro de un nn.Module
        # (como lo que tenemos) automáticamente funciona como parámetros.        
        self.factors = nn.Parameter(torch.Tensor(nt, nx, nz))
        # En este caso la función dinámica es nada más y nada menos que un MLP, que en el caso sencillo es solo una transfor
        # mación lienal, tal y como comenta en el paper.
        # Lo único es que esta vez la entrada es de más de una variable, dependiendo de nz y nr. La salida sigue siendo nz.
#        self.dynamic = MLP(nz * self.nr + np * 2, nhid, nz, nlayers, dropout_d)
        self.dynamic = MLP(nz * self.nr + np * 4, nhid, nz, nlayers, dropout_d)
        # El famoso decoder. En el paper no se aportan expresiones para él, por lo que no tengo muy claro porque se ha decido
        # por este en concreto. 
        # En cualquier caso, nn.Linear aplica una transformación lineal de un espacio de nz dimensiones a otro de nd. Funciona
        # como una función, es decir aquí estamos definiendo decoder como un transformador lineal de nz a nd.
        # Por ejemplo con el factors creado, si haces decoder(factors[-1]) devuelve un tensor de 49x49 (espacio real)
        self.decoder = nn.Linear(nz, nd, bias=False)
        if mode == 'refine':
            self.relations.data = self.relations.data.ceil().clamp(0, 1).byte()
            self.rel_weights = nn.Parameter(torch.Tensor(self.relations.sum().item() - self.nx))
        elif mode == 'discover':
            self.rel_weights = nn.Parameter(torch.Tensor(nx, 1, nx))
        # init
        self._init_weights(periode)

    def _init_weights(self, periode):
        initrange = 0.1
        if periode >= self.nt:
            self.factors.data.uniform_(-initrange, initrange)
        else:
            timesteps = torch.arange(self.factors.size(0)).long()
            for t in range(periode):
                idx = timesteps % periode == t
                idx_data = idx.view(-1, 1, 1).expand_as(self.factors)
                init = torch.Tensor(self.nx, self.nz).uniform_(-initrange, initrange).repeat(idx.sum().item(), 1, 1)
            self.factors.data.masked_scatter_(idx_data, init.view(-1))
        if self.mode == 'refine':
            self.rel_weights.data.fill_(0.5)
        elif self.mode == 'discover':
            self.rel_weights.data.fill_(1 / self.nx)

    def get_relations(self):
        if self.mode is None:
            return self.relations
        else:
            weights = F.hardtanh(self.rel_weights, 0, 1)
            if self.mode == 'refine':
                intra = self.rel_weights.new(self.nx, self.nx).copy_(self.relations[:, 0]).unsqueeze(1)
                inter = self.rel_weights.new_zeros(self.nx, self.nr - 1, self.nx)
                inter.masked_scatter_(self.relations[:, 1:], weights)
            if self.mode == 'discover':
                intra = self.relations[:, 0].unsqueeze(1)
                inter = weights
            return torch.cat((intra, inter), 1)
        
    def update_z(self, z, exogenous_var):
        """
        Dado un conjunto de Zs, calcula Zt+1. Presupone la existencia
        de una función dinámica ya entrenada, y de la entrada de variables exógenas
        en diversos rangos temporales.
        MÁS DE UN RANGO TEMPORAL
        :param z: el valor de las variables en el espacio latente
        :return Zt+1: el valor que toma el espacio latente en el siguiente tiempo
        """
        z_context = self.get_relations().matmul(z).view(-1, self.nr * self.nz)
        if self.np != 0:
            exo_context_0 = self.get_relations().matmul(exogenous_var[:,0]).view(-1, self.nr * self.np)
            exo_context_1 = self.get_relations().matmul(exogenous_var[:,1]).view(-1, self.nr * self.np)
            z_cat = torch.cat((z_context, exo_context_0, exo_context_1),1)
#          z_next = self.dynamic(z_context)
            z_next = self.dynamic(z_cat)
            return self.activation(z_next)
        else:
            z_next = self.dynamic(z_context)
            return self.activation(z_next)   

#    def update_z(self, z, exogenous_var):
#        """
#        Dado un conjunto de Zs, calcula Zt+1. Presupone la existencia
#        de una función dinámica ya entrenada.
#        :param z: el valor de las variables en el espacio latente
#        :return Zt+1: el valor que toma el espacio latente en el siguiente tiempo
#        """
#        z_context = self.get_relations().matmul(z).view(-1, self.nr * self.nz)
#        if self.np != 0:
#            exo_context = self.get_relations().matmul(exogenous_var).view(-1, self.nr * self.np)
#            z_cat = torch.cat((z_context, exo_context),1)
##          z_next = self.dynamic(z_context)
#            z_next = self.dynamic(z_cat)
#            return self.activation(z_next)
#        else:
#            z_next = self.dynamic(z_context)
#            return self.activation(z_next)
             

    def decode_z(self, z):
        """
        Dado un conjunto de Zs, hace la decodificación. Presupone la existencia
        de un decodificador ya entrenado.
        :param z: el valor de las variables en el espacio latente
        :return x_rec: el valor que toma la salida en el espacio real
        """
        x_rec = self.decoder(z)
        return x_rec

    def dec_closure(self, t_idx, x_idx):
        """
        "Entrenamiento" del decodificador (d en el paper). Realmente no existe un entrenamiento
        :param t_idx: índice de la serie temporal
        :param x_idx: índice de la serie espacial
        """
        # Esto básicamente coge un elemento concreto de factors (ún número, lista o lo que sea. Una fila entera, según la
        # dimensión de nz), y hace con cierta probabilidad 0 componentes suyos con drop.
        z_inf = self.drop(self.factors[t_idx, x_idx])
        # Ahora es cuando hace la parte del decoder (linear)
        x_rec = self.decoder(z_inf)
        return x_rec

    
    def dyn_closure(self, t_idx, x_idx):
        """
        "Entrenamiento" de la función dinámica (g en el paper). Realmente no existe un entrenamiento.
        MÁS DE UN RANGO TEMPORAL
        :param t_idx: índices de la serie temporal
        :param x_idx: índices de la serie en sí
        """
        # Se obtienen las relaciones.
        rels = self.get_relations()
        # Y los factors que toquen, con la posibilidad de ponerlos a 0.
        z_input = self.drop(self.factors[t_idx])
        # matmul es un producto matricial de tensores (se puede poner como torch.matmul(tensor, tensor2) o como tensor1.matmul(tensor2))
        # view devuelve un tensor con las mismas componentes que el inicial, pero con diferente shape. En este caso, pasa de
        # columna a fila. Para hacerlos una idea, es como que saca una fila de dos elementos, uno con la Z anterior y otro 
        # con el producto escalar de la fila de W que toque con las Z anteriores para su uso en la ecuación (4). Si Z tiene más
        # dimensión que uno, pues serán más componentes.
        z_context = rels[x_idx].matmul(z_input).view(-1, self.nr * self.nz)
        if self.np != 0:
#            perm = torch.LongTensor([torch.Tensor.numpy(t_idx + 1), torch.Tensor.numpy(x_idx)])
#            exo = self.exogenous[perm[0], perm[1]] SIN W EN EXOGENAS
#            exo = self.exogenous[perm[0]]
            exo_input_1 = self.exogenous[t_idx+1]
            exo_input_0 = self.exogenous[t_idx]
            exo_context_1 = rels[x_idx].matmul(exo_input_1).view(-1, self.nr * self.np)
            exo_context_0 = rels[x_idx].matmul(exo_input_0).view(-1, self.nr * self.np)
            z_cat = torch.cat((z_context, exo_context_0, exo_context_1),1)
            z_gen = self.dynamic(z_cat)
            return self.activation(z_gen)
        else:
            z_gen = self.dynamic(z_context)
            return self.activation(z_gen)
        
#        def dyn_closure(self, t_idx, x_idx):
#        """
#        "Entrenamiento" de la función dinámica (g en el paper). Realmente no existe un entrenamiento
#        :param t_idx: índices de la serie temporal
#        :param x_idx: índices de la serie en sí
#        """
#        # Se obtienen las relaciones.
#        rels = self.get_relations()
#        # Y los factors que toquen, con la posibilidad de ponerlos a 0.
#        z_input = self.drop(self.factors[t_idx])
#        # matmul es un producto matricial de tensores (se puede poner como torch.matmul(tensor, tensor2) o como tensor1.matmul(tensor2))
#        # view devuelve un tensor con las mismas componentes que el inicial, pero con diferente shape. En este caso, pasa de
#        # columna a fila. Para hacerlos una idea, es como que saca una fila de dos elementos, uno con la Z anterior y otro 
#        # con el producto escalar de la fila de W que toque con las Z anteriores para su uso en la ecuación (4). Si Z tiene más
#        # dimensión que uno, pues serán más componentes.
#        z_context = rels[x_idx].matmul(z_input).view(-1, self.nr * self.nz)
#        if self.np != 0:
##            perm = torch.LongTensor([torch.Tensor.numpy(t_idx + 1), torch.Tensor.numpy(x_idx)])
##            exo = self.exogenous[perm[0], perm[1]] SIN W EN EXOGENAS
##            exo = self.exogenous[perm[0]]
#            exo_input = self.exogenous[t_idx+1]
#            exo_context = rels[x_idx].matmul(exo_input).view(-1, self.nr * self.np)
#            z_cat = torch.cat((z_context, exo_context),1)
#            z_gen = self.dynamic(z_cat)
#            return self.activation(z_gen)
#        else:
#            z_gen = self.dynamic(z_context)
#            return self.activation(z_gen)
        
        
    def generate(self, nsteps, validation_exo):
        """
        Función que genera una salida para la variable de estudio y para la función
        Z del latent space. Lo hace para un número nsteps de pasos.
        MÁS DE UN RANGO TEMPORAL
        :param nsteps: número de pasos temporales para los cuales se calculan las variables
        """
        # Se coge el valor de z del último de los ejemplos de entrenamiento parece ser (todo esto suponiendo que factors es
        # lo que estoy suponiendo que es). A partir de él se calcularán el primero nuevo, a partir de este el segundo, y así.        
        z = self.factors[-1]
        ex = self.exogenous[-1]
        z_gen = []
        for t in range(nsteps):
            z = self.update_z(z, torch.cat((ex, validation_exo[t]), 1))
            ex = validation_exo[t]
            z_gen.append(z)
#        CON Zt+1 - LAMBDAt
#        for t in range(nsteps):
#            if t == 0:
#                z = self.update_z(z, ex)
#            else:
#                z = self.update_z(z, validation_exo[t-1])
#            z_gen.append(z)
        # Concatenación de tensores. Es decir, z_gen es una lista de tensores (cada update_z da uno), aquí se genera un único
        # tensor largo.
        # Prueba por ejemplo: torch.stack([torch.tensor([[1,2],[3,4]]), torch.tensor([[5,6],[7,8]])])
        z_gen = torch.stack(z_gen)
        # Esto simplemente calcula las xs para esos valores de z.
        x_gen = self.decode_z(z_gen)
        return x_gen, z_gen    
        
#    def generate(self, nsteps, validation_exo):
#        """
#        Función que genera una salida para la variable de estudio y para la función
#        Z del latent space. Lo hace para un número nsteps de pasos.
#        :param nsteps: número de pasos temporales para los cuales se calculan las variables
#        """
#        # Se coge el valor de z del último de los ejemplos de entrenamiento parece ser (todo esto suponiendo que factors es
#        # lo que estoy suponiendo que es). A partir de él se calcularán el primero nuevo, a partir de este el segundo, y así.        
#        z = self.factors[-1]
##        ex = self.exogenous[-1]
#        z_gen = []
#        for t in range(nsteps):
#            z = self.update_z(z, validation_exo[t])
#            z_gen.append(z)
##        CON Zt+1 - LAMBDAt
##        for t in range(nsteps):
##            if t == 0:
##                z = self.update_z(z, ex)
##            else:
##                z = self.update_z(z, validation_exo[t-1])
##            z_gen.append(z)
#        # Concatenación de tensores. Es decir, z_gen es una lista de tensores (cada update_z da uno), aquí se genera un único
#        # tensor largo.
#        # Prueba por ejemplo: torch.stack([torch.tensor([[1,2],[3,4]]), torch.tensor([[5,6],[7,8]])])
#        z_gen = torch.stack(z_gen)
#        # Esto simplemente calcula las xs para esos valores de z.
#        x_gen = self.decode_z(z_gen)
#        return x_gen, z_gen

    def factors_parameters(self):
        yield self.factors

    def rel_parameters(self):
        assert self.mode is not None
        yield self.rel_weights
