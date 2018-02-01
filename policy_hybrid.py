import numpy as np


class HybridLinUCB():
	def __init__(self):
		self.alpha = 2.1 # 1 + np.sqrt(np.log(2/delta)/2)
		self.r1 = 0.8
		self.r0 = -20
		self.d = 6 # dimension of user features = d
		self.k = self.d * self.d # dimension of article features = k
		self.article_features = {}
		self.A0 = np.identity(self.k) # A0 : matrix to compute hybrid part, k*k
		self.A0I = np.identity(self.k) # A0I: inverse of A0
		self.b0 = np.zeros((self.k, 1)) # b0 : vector to compute hybrid part, k
		self.Aa = {} # Aa : collection of matrix to compute disjoint part for each article a, d*d
		self.AaI = {} # AaI : collection of matrix to compute disjoint part for each article a, d*d
		self.Ba = {} # Ba : collection of matrix to compute hybrid part, d*k
		self.BaT = {} # BaT : collection of matrix to compute hybrid part, d*k
		self.ba = {} # ba : collection of vectors to compute disjoin part, d*1
		self.AaIba = {}
		self.AaIBa = {}
		self.A0IBaTAaI = {}
		# self.AaIBaA0IBaTAaI = {}
		self.theta = {}
		self.beta = np.zeros((self.k, 1))
		self.index_all = {}
		self.a_max = None
		self.z = None
		self.zT = None
		self.xaT = None
		self.xa = None

	def set_articles(self, articles):
		i = 0
		art_len = len(articles)
		self.article_features = np.zeros((art_len, 1, self.d))
		self.Aa = np.zeros((art_len, self.d, self.d))
		self.AaI = np.zeros((art_len, self.d, self.d))
		self.Ba = np.zeros((art_len, self.d, self.k))
		self.BaT = np.zeros((art_len, self.k, self.d))
		self.ba = np.zeros((art_len, self.d, 1))
		self.AaIba = np.zeros((art_len, self.d, 1))
		self.AaIBa = np.zeros((art_len, self.d, self.k))
		self.A0IBaTAaI = np.zeros((art_len, self.k, self.d))
		# self.AaIBaA0IBaTAaI = np.zeros((art_len, self.d, self.d))
		self.theta = np.zeros((art_len, self.d, 1))
		for key in articles:
			self.index_all[key] = i
			self.article_features[i] = articles[key][:]
			self.Aa[i] = np.identity(self.d)
			self.AaI[i] = np.identity(self.d)
			self.Ba[i] = np.zeros((self.d, self.k))
			self.BaT[i] = np.zeros((self.k, self.d))
			self.ba[i] = np.zeros((self.d, 1))
			self.AaIba[i] = np.zeros((self.d, 1))
			self.AaIBa[i] = np.zeros((self.d, self.k))
			self.A0IBaTAaI[i] = np.zeros((self.k, self.d))
			# self.AaIBaA0IBaTAaI[i] = np.zeros((self.d, self.d))
			self.theta[i] = np.zeros((self.d, 1))
			i += 1


	def update(self, reward):
		if reward == -1:
			pass
		elif reward == 1 or reward == 0:
			if reward == 1:
			    r = self.r1
			else:
			    r = self.r0

			self.A0 += self.BaT[self.a_max].dot(self.AaIBa[self.a_max])
			self.b0 += self.BaT[self.a_max].dot(self.AaIba[self.a_max])
			self.Aa[self.a_max] += np.dot(self.xa, self.xaT)
			self.AaI[self.a_max] = np.linalg.inv(self.Aa[self.a_max])
			self.Ba[self.a_max] += np.dot(self.xa, self.zT)
			self.BaT[self.a_max] = np.transpose(self.Ba[self.a_max])
			self.ba[self.a_max] += r * self.xa
			self.AaIba[self.a_max] = np.dot(self.AaI[self.a_max], self.ba[self.a_max])
			self.AaIBa[self.a_max] = np.dot(self.AaI[self.a_max], self.Ba[self.a_max])

			self.A0 += np.dot(self.z, self.zT) - np.dot(self.BaT[self.a_max], self.AaIBa[self.a_max])
			self.b0 += r * self.z - np.dot(self.BaT[self.a_max], self.AaIba[self.a_max])
			self.A0I = np.linalg.inv(self.A0)
			self.A0IBaTAaI[self.a_max] = self.A0I.dot(self.BaT[self.a_max]).dot(self.AaI[self.a_max])
			# self.AaIBaA0IBaTAaI[self.a_max] = np.matmul(self.AaIBa[self.a_max], self.A0IBaTAaI[self.a_max])
			self.beta = np.dot(self.A0I, self.b0)
			self.theta = self.AaIba - np.dot(self.AaIBa, self.beta)

		else:
			pass


	def recommend(self, timestamp, user_features, articles):
		article_len = len(articles) # 20

		self.xa = np.array(user_features).reshape((self.d,1)) # (6,1)
		self.xaT = np.transpose(self.xa) # (1,6)

		index = [self.index_all[article] for article in articles]
		article_features_tmp = self.article_features[index]

		# za : feature of current user/article combination, k*1
		za = np.outer(article_features_tmp.reshape(-1), self.xa).reshape((article_len,self.k,1)) # (20,36,1)
		zaT = np.transpose(za, (0,2,1)) # (20,1,36)

		A0Iza = np.matmul(self.A0I, za) # (20,36,1)
		A0IBaTAaIxa = np.matmul(self.A0IBaTAaI[index], self.xa) # (20,36,1)
		AaIxa = self.AaI[index].dot(self.xa) # (20,6,1)
		AaIBaA0IBaTAaIxa = np.matmul(self.AaIBa[index], A0IBaTAaIxa) # (20,6,1)
		# AaIBaA0IBaTAaIxa = np.matmul(self.AaIBaA0IBaTAaI[index], self.xa) # (20,6,1)

		s = np.matmul(zaT, A0Iza - 2*A0IBaTAaIxa) + np.matmul(self.xaT, AaIxa + AaIBaA0IBaTAaIxa) # (20,1,1)
		p = zaT.dot(self.beta) + np.matmul(self.xaT, self.theta[index]) + self.alpha*np.sqrt(s) # (20,1,1)
		# assert (s < 0).any() == False
		# assert np.isnan(np.sqrt(s)).any() == False

		# print A0Iza.shape, A0IBaTAaIxa.shape, AaIxa.shape, AaIBaA0IBaTAaIxa.shape, s.shape, p.shape (for debugging)
		max_index = np.argmax(p)
		self.z = za[max_index]
		self.zT = zaT[max_index]
		art_max = index[max_index]
		self.a_max = art_max # article index with largest UCB

		return articles[max_index]


def set_articles(articles):
	global HybridLinUCB
	HybridLinUCB = HybridLinUCB()
	HybridLinUCB.set_articles(articles)

def update(reward):
	return HybridLinUCB.update(reward)

def recommend(timestamp, user_features, articles):
	return HybridLinUCB.recommend(timestamp, user_features, articles)
