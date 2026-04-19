import networkx

from .network import G

# TODO: Hub centrality thingy analysis. Summary:
# We hypothesize that betweenness centrality is concentrated in a small number of interchange hubs 
# (places like Chicago, New York Penn Station, Washington Union Station), and that removing the top five-ish stations 
# by betweenness would reduce the size of the largest connected component far more severely than removing a similar number 
# of random stations. If true, this would mean that the Amtrak system is vulnerable to coordinated attack, 
# but not so much to random failure — it would also tell us that the five-ish most important stations really are very important 
# relative to all the others.


