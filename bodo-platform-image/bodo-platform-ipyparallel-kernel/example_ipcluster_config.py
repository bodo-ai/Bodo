import psutil

c.Cluster.engine_launcher_class = "MPI"
c.Cluster.n = psutil.cpu_count(logical=False)
c.Cluster.controller_ip = "*"
c.Cluster.controller_args = ["--nodb"]
