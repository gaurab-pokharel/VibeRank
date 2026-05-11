import numpy as np
class GraphTools:
	def __init__(self,edges):
		# edges: a list of pairs (u,v) signifying edge u->v; we assume u and v are strings that uniquely identify the nodes
		# init transforms this into an adjacency list and stores the information here
		self.edges = edges
		self.nodes = set()
		self.adj_list = {}
		
		for u,v in edges:
			self.nodes.add(u)
			self.nodes.add(v)
			self.adj_list.setdefault(u,set()).add(v)
			if v not in self.adj_list:
				self.adj_list[v] =  set()
		

		self.out_degrees = {node:len(self.adj_list[node]) for node in self.nodes}

		return
	
	def validate_complete_graph(self):
		# debugging tool to validate the graph is, in fact, a complete graph (if it were undirected)
		# idea, unique number of unordered edges will be n(n-1)/2
		uniq_edge_set = set()
		for u,v in self.edges:
			uniq_edge_set.add(tuple(sorted([u, v])))
		n = len(self.nodes)
		if len(uniq_edge_set)==n*(n-1)/2:
			return "passed"
		return "failed"



	def nc2(self, n):
		return n*(n-1)//2
	
	def nc3(self,n):
		# n!/3!(n-3)! = n*(n-1)*(n-2)/6
		return n*(n-1)*(n-2)//6

	def count_triads(self):

		
		
		triads = self.nc3(len(self.nodes)) - np.sum([self.nc2(self.out_degrees[node]) for node in self.nodes])

		return triads
	
	def naive_count_triads(self):
		def order_cycle(u, v, w):
			return tuple(sorted([u, v, w]))

		cycles = set()

		for u in self.nodes:
			for v in self.adj_list.get(u, set()):
				if u == v:
					continue

				for w in self.adj_list.get(v, set()):
					if w == u or w == v:
						continue

					# Check w -> u to close the cycle
					if u in self.adj_list.get(w, set()):
						cycles.add(order_cycle(u, v, w))

		return len(cycles), cycles
