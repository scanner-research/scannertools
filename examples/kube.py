from scannertools import kube

cloud_config = kube.CloudConfig(
    project='visualdb-1046',
    service_key='',
    storage_key_id='',
    storage_key_secret='')

cluster_config = kube.ClusterConfig(
    id='wc-test',
    num_workers=1)

cluster = kube.Cluster(cloud_config, cluster_config)

cluster.cli()
