
import random

# All of the following code provided in MIT 6.036 Intro. to Machine Learning
# which includes DDist probability distribution class with 2 additional functions uniform_dist and delta_dist

class DDist:
    """Discrete distribution represented as a dictionary.  Can be
    sparse, in the sense that elements that are not explicitly
    contained in the dictionary are assuemd to have zero probability."""
    def __init__(self, dictionary, name = None):
        self.d = dictionary
        """ Dictionary whose keys are elements of the domain and values
        are their probabilities. """

    def prob(self, elt):
        """
        @returns: the probability associated with C{elt}
        """
        return self.d.get(elt, 0)

    def setProb(self, elt, p):
        """
        @param elt: element of the domain
        @param p: probability
        Sets probability of C{elt} to be C{p}
        """
        self.d[elt] = p

    def support(self):
        """
        @returns: A list (in any order) of the elements of this
        distribution with non-zero probabability.
        """
        return self.d.keys()

    def maxProbElt(self):
        """
        @returns: The element in this domain with maximum probability
        """
        bestP = 0
        bestElt = None
        for (elt, p) in self.d.items():
            if p > bestP:
                bestP = p
                bestElt = elt
        return (bestElt, bestP)

    def draw(self):
        """
        @returns: a randomly drawn element from the distribution
        """
        r = random.random()
        sum = 0.0
        for val in self.support():
            sum += self.prob(val)
            if r < sum:
                return val
        raise Exception('Failed to draw from '+ str(self))

    def addProb(self, val, p):
        """
        Increase the probability of element C{val} by C{p}
        """
        self.setProb(val, self.prob(val) + p)

    def mulProb(self, val, p):
        """
        Multiply the probability of element C{val} by C{p}
        """
        self.setProb(val, self.prob(val) * p)

    def expectation(self, f):
        return sum(self.prob(x) * f(x) for x in self.support())

    def normalize(self):
        """
        Divides all probabilities through by the sum of the values to
        ensure the distribution is normalized.

        Changes the distribution!!  (And returns it, for good measure)

        Generates an error if the sum of the current probability
        values is zero.
        """
        z = sum([self.prob(e) for e in self.support()])
        assert z > 0.0, 'degenerate distribution ' + str(self)
        alpha = 1.0 / z
        for e in self.support():
            self.mulProb(e, alpha)
        return self

def uniform_dist(elts):
    """
    Uniform distribution over a given finite set of C{elts}
    @param elts: list of any kind of item
    """
    p = 1.0 / len(elts)
    return DDist(dict([(e, p) for e in elts]))    

def delta_dist(elt):
    return DDist({elt: 1.0})
