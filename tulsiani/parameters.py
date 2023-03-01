import torch

class TulsianiParams:
    @property
    def dims_factor(self):
        return 1 if self.use_paschalidou_loss else self.dims_factors[self.phase]

    @property
    def prob_factor(self):
        return 1 if self.use_paschalidou_loss else self.prob_factors[self.phase]

    @property
    def existence_penalty(self):
        return self.existence_penalties[self.phase]

    @property
    def iterations(self):
        if self.use_paschalidou_loss:
            return self.paschalidou_n_iterations
        else:
            return self.n_iterations[self.phase]

    @property
    def total_iterations(self):
        if self.use_paschalidou_loss:
            return self.paschalidou_n_iterations
        else:
            return sum(self.n_iterations)

params = TulsianiParams()

params.train_dir = 'data/train'
params.urocell_dir = 'data/urocell'

# Ali naj imamo sploh ločeno učno, validacijsko in testno množico?
params.use_split = True

# Število iteracij treniranja v prvi in drugi fazi:
params.n_iterations = [20000, 30000]
# Toliko iteracij se vsak batch ponovi (t.j. po tolikšnem številu
# naložimo nov batch), glej 'params.modelIter' v referenčni implementaciji.
# (Bolj strogo gledano se na toliko iteracij ponovi batch z istimi modeli,
# ker potem še za vsako iteracijo vzorčimo podmnožico točk na površini oblike.)
params.repeat_batch_n_iterations = 2
# Na vsake toliko iteracij se shrani model, če je validacijski loss manjši:
params.save_iteration = 1000

# Ali napovedujemo prisotnost primitivov (True) ali pa kar vedno vzamemo vse (False):
params.prune_primitives = True
params.n_primitives = 20
# Iz vsake oblike smo med predprocesiranjem vzorčili 10.000 točk,
# vgendar naenkrat bomo upoštevali samo 1000 naključnih točk:
params.n_samples_per_shape = 1000
params.n_samples_per_primitive = 150

params.iou_n_points = 10000

params.batch_size = 32
params.use_batch_normalization_conv = True
params.use_batch_normalization_linear = True
params.add_coordinates_to_encoder = True
params.learning_rate = 1e-3
params.reinforce_baseline_momentum = .9

# S faktorjem 'dims_factor' dosežemo, da je aktivacija za napovedovanje
# dimenzij bolj "razpotegnjena" (dims = sigmoid(dims_factor * features)).
# Tako kompenziramo nagnjenje metode k temu, da slabo ujemajoče kvadre
# samo zmanjša namesto, da bi poiskala boljše translacije in rotacije.
# Z drugimi besedami: dimenzij se učimo počasneje kot translacije in rotacije
# (https://github.com/nileshkulkarni/volumetricPrimitivesPytorch/issues/3).
# Vrednost tega faktorja je večja v drugi fazi, kjer problem zmanjšanja
# slabo ujemajočih kvadrov ni tako izrazit, saj se jim lahko samo zmanjša
# verjetnost.
params.dims_factors = [0.01, 0.5] # prva in druga faza

# Ta faktor ima isto nalogo, ampak je za napovedovanje verjetnosti.
# V prvi fazi učenja je faktor zelo majhen, saj poskusimo optimizirati ostale
# parametre, preden dovolimo, da se nad kvadrom "obupa".
params.prob_factors = [0.0001, 0.2]

# V drugi fazi tudi rahlo penaliziramo prisotnost.
params.existence_penalties = [.0, 8e-5]

# Ko je 'use_chamfer = True', polj razdalj v notranjosti ne nastavimo na nič
# (enako kot isto imenska opcija pri metodi Paschalidou). Paschalidou in sod. 
# so ugotovili, da tako zaidemo v manj lokalnih minimumov...
params.use_chamfer = False

# Tega ne uporabljamo:
params.use_paschalidou_loss = False
params.paschalidou_n_iterations = 15000
params.paschalidou_alpha = 1.
params.paschalidou_beta = 1e-3

params.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
