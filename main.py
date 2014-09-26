import numpy as np
import theano
from theano import tensor as tt
from theano.compat import OrderedDict


def change_beta(beta, w, K, N):
    ''''''
    # beta: K x ..
    # w: N
    beta_prime = np.zeros(K, N)
    for n in xrange(N):
        beta_prime[:, n] = beta[:, w[n]]

    return beta_prime


def init_phi(K, N):
    return np.ones((N, K)) / float(K)


def init_gamma(alpha, K, N):
    return alpha * np.ones(K) + N / float(K)


def main():
    print "hi"
    ALPHA = 0.01
    N = 100
    K = 10
    V = 1000
    M = 3
    EPS = 0.01
    w = tt.ivector("w")
    beta = tt.fmatrix("beta")

    phi = theano.shared(init_phi(K, N), "phi")
    gamma = theano.shared(init_gamma(ALPHA, K, N), "gamma")
    beta_prime = tt.matrix("beta_prime")
    alpha = tt.vector("alpha")

    phi_update = (beta_prime * tt.exp(tt.psi(gamma)).dimshuffle(0, "x")).T # N x K
    updates = OrderedDict()
    new_phi = phi_update / tt.sum(phi_update, axis=1).dimshuffle(0, "x")
    updates[phi] = new_phi

    updates[gamma] = alpha + tt.sum(new_phi, axis=0)

    e_step = theano.function([beta_prime, alpha], [], updates=updates)

    #m_step = theano.function([w], [], updates=)
    # w_value = np.random.randint(0, N - 1, N * k).reshape(k, N)
    #w_value.dtype = "int32"
    alpha_value = ALPHA
    beta_prime_value = np.arange(0.0, N * K, dtype="float32").reshape(K, N)

    phi_prev = None
    gamma_prev = None
    i = 0
    while (phi_prev is None and gamma_prev is None) or ((np.abs(phi.get_value() - phi_prev) < EPS).all()
                                                        and (np.abs(gamma.get_value() - gamma_prev) < EPS).all()):
        i += 1
        if i % 1000 == 0:
            print i
        e_step(beta_prime=beta_prime_value, alpha=np.repeat(ALPHA, K))
        phi_prev = phi.get_value()
        gamma_prev = gamma.get_value()

    print "Result", res, "phi", phi.get_value()


def gibbs():
    N = 100 # max words
    K = 10 # topics
    W = 1000 # vocabulary

    NWKJ = tt.TensorType("float32", broadcastable=(False, False, False), name="NWKJ")

    trng = tt.shared_randomstreams.RandomStreams(1235)

    u = trng.uniform()
    p = NWKJ
    print theano.function([], d)()


if __name__ == "__main__":
    #main()
    gibbs()