from attr import attrs, attrib
import subprocess as sp
import time
import json
import yaml
import tempfile
import argparse

def run(s):
    return sp.check_output(s, shell=True)


@attrs
class CloudConfig:
    project = attrib(type=str)
    service_key = attrib(type=str)
    storage_key_id = attrib(type=str)
    storage_key_secret = attrib(type=str)


@attrs
class MachineConfig:
    cpu = attrib(type=int)
    mem = attrib(type=int)
    disk = attrib(type=int)
    gpu = attrib(type=int, default=0)
    gpu_type = attrib(type=str, default='nvidia-tesla-p100')

    def type_name(self):
        name = 'custom-{}-{}'.format(self.cpu, self.mem * 1024)
        if float(self.mem) / self.cpu > 6.5:
            name += '-ext'
        return name


@attrs
class ClusterConfig:
    id = attrib(type=str)
    num_workers = attrib(type=int)
    zone = attrib(type=str, default='us-east1-d')
    kube_version = attrib(type=str, default='1.9.7-gke.3')
    master = attrib(type=MachineConfig, default=MachineConfig(cpu=4, mem=96, disk=50))
    worker = attrib(type=MachineConfig, default=MachineConfig(cpu=64, mem=256, disk=500))
    workers_per_node = attrib(type=int, default=1)
    preemptible = attrib(type=bool, default=False)
    autoscale = attrib(type=bool, default=False)
    scopes = attrib(
        type=list,
        default=[
            "https://www.googleapis.com/auth/compute",
            "https://www.googleapis.com/auth/devstorage.read_write",
            "https://www.googleapis.com/auth/logging.write",
            "https://www.googleapis.com/auth/monitoring", "https://www.googleapis.com/auth/pubsub",
            "https://www.googleapis.com/auth/servicecontrol",
            "https://www.googleapis.com/auth/service.management.readonly",
            "https://www.googleapis.com/auth/trace.append"
        ])


class Cluster:
    def __init__(self, cloud_config, cluster_config):
        self._cloud_config = cloud_config
        self._cluster_config = cluster_config
        self._cluster_cmd = 'gcloud container --project {} clusters --zone {}' \
            .format(self._cloud_config.project, self._cluster_config.zone)

    def get_kube_info(self, kind, namespace='default'):
        return json.loads(
            run('kubectl get {} -o json -n {}'.format(kind, namespace)).decode('utf-8'))

    def get_by_owner(self, ty, owner, namespace='default'):
        return run(
            'kubectl get {} -o json -n {} | jq \'.items[] | select(.metadata.ownerReferences[0].name == "{}") | .metadata.name\'' \
            .format(ty, namespace, owner)) \
            .decode('utf-8').strip()[1:-1]

    def get_object(self, info, name):
        for item in info['items']:
            if item['metadata']['name'] == name:
                return item
        return None

    def get_pod(self, deployment, namespace='default'):
        while True:
            rs = self.get_by_owner('rs', deployment, namespace)
            pod_name = self.get_by_owner('pod', rs, namespace)
            if "\n" not in pod_name and pod_name != "":
                break
            time.sleep(1)

        while True:
            pod = self.get_object(self.get_kube_info('pod', namespace), pod_name)
            if pod is not None:
                return pod
            time.sleep(1)

    def running(self):
        return run('{cmd} list --format=json | jq \'.[] | select(.name == "{id}")\''.format(
            cmd=self._cluster_cmd, id=self._cluster_config.id)) != b''

    def machine_start(self):
        cfg = self._cluster_config
        fmt_args = {
            'cmd': self._cluster_cmd,
            'cluster_id': cfg.id,
            'cluster_version': cfg.kube_version,
            'master_machine': cfg.master.type_name(),
            'master_disk': cfg.master.disk,
            'worker_machine': cfg.worker.type_name(),
            'worker_disk': cfg.worker.disk,
            'scopes': ','.join(cfg.scopes),
            'initial_size': cfg.num_workers,
            'accelerator': '--accelerator type={},count={}'.format(cfg.worker.gpu_type, cfg.worker.gpu) if cfg.worker.gpu > 0 else '',
            'preemptible': '--preemptible' if cfg.preemptible else '',
            'autoscaling': '--enable-autoscaling --min-nodes 0 --max-nodes {}'.format(cfg.num_workers) if cfg.autoscale else ''
        }  # yapf: disable

        cluster_cmd = """
{cmd} -q create "{cluster_id}" \
        --enable-kubernetes-alpha \
        --cluster-version "{cluster_version}" \
        --machine-type "{master_machine}" \
        --image-type "COS" \
        --disk-size "{master_disk}" \
        --scopes {scopes} \
        --num-nodes "1" \
        --enable-cloud-logging \
        {accelerator}
        """.format(**fmt_args)

        run(cluster_cmd)

        fmt_args['cmd'] = fmt_args['cmd'].replace('clusters', 'node-pools')
        pool_cmd = """
{cmd} -q create workers \
        --cluster "{cluster_id}" \
        --machine-type "{worker_machine}" \
        --image-type "COS" \
        --disk-size "{worker_disk}" \
        --scopes {scopes} \
        --num-nodes "{initial_size}" \
        {autoscaling} \
        {preemptible} \
        {accelerator}
        """.format(**fmt_args)

        run(pool_cmd)

        # Wait for cluster to enter reconciliation if it's going to occur
        if cfg.num_workers > 1:
            time.sleep(60)

        # If we requested workers up front, we have to wait for the cluster to reconcile while
        # they are being allocated
        while True:
            cluster_status = run(
                '{cmd} list --format=json | jq -r \'.[] | select(.name == "{id}") | .status\''.
                format(cmd=self._cluster_cmd, id=cfg.id)).strip().decode('utf-8')

            if cluster_status == 'RECONCILING':
                time.sleep(5)
            else:
                if cluster_status != 'RUNNING':
                    raise Exception(
                        'Expected cluster status RUNNING, got: {}'.format(cluster_status))
                break

        if cfg.worker.gpu > 0:
            # Install GPU drivers
            # https://cloud.google.com/kubernetes-engine/docs/concepts/gpus#installing_drivers
            run('kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/stable/nvidia-driver-installer/cos/daemonset-preloaded.yaml'
                )

        print(
            'https://console.cloud.google.com/kubernetes/clusters/details/{zone}/{cluster_id}?project={project}&tab=details' \
            .format(zone=cfg.zone, project=self._cloud_config.project, **fmt_args))

    def get_credentials(self):
        run('{cmd} get-credentials {id}'.format(cmd=self._cluster_cmd, id=self._cluster_config.id))

    def create_object(self, template):
        with tempfile.NamedTemporaryFile() as f:
            f.write(yaml.dump(template).encode())
            f.flush()
            run('kubectl create -f {}'.format(f.name))

    def make_container(self, name):
        template = {
            'name': name,
            'image': 'gcr.io/{project}/scanner-{name}:{device}'.format(
                project=self._cluster_config.id,
                name=name,
                device='gpu' if self._cluster_config.worker.gpu > 0 else 'cpu'),
            'imagePullPolicy': 'Always',
            'volumeMounts': [{
                'name': 'service-key',
                'mountPath': '/secret'
            }],
            'env': [
                {'name': 'GOOGLE_APPLICATION_CREDENTIALS',
                 'value': '/secret/service-key.json'},
                {'name': 'AWS_ACCESS_KEY_ID',
                 'valueFrom': {'secretKeyRef': {
                     'name': 'aws-storage-key',
                     'key': 'AWS_ACCESS_KEY_ID'
                 }}},
                {'name': 'AWS_SECRET_ACCESS_KEY',
                 'valueFrom': {'secretKeyRef': {
                     'name': 'aws-storage-key',
                     'key': 'AWS_SECRET_ACCESS_KEY'
                 }}},
                {'name': 'GLOG_minloglevel',
                 'value': '0'},
                {'name': 'GLOG_logtostderr',
                 'value': '1'},
                {'name': 'GLOG_v',
                 'value': '1'},
                {'name': 'WORKERS_PER_NODE',
                 'value': str(self._cluster_config.workers_per_node)},
                {'name': 'DEPS',
                 'value': ','.join([])}, # TODO(wcrichto)
                # HACK(wcrichto): GPU decode for interlaced videos is broken, so forcing CPU
                # decode instead for now.
                {'name': 'FORCE_CPU_DECODE',
                 'value': '1'}
            ],
            'resources': {},
            'securityContext': {'capabilities': {
                'add': ['SYS_PTRACE']  # Allows gdb to work in container
            }}
        }  # yapf: disable
        if name == 'master':
            template['ports'] = [{
                'containerPort': 8080,
            }]

        if self._cluster_config.worker.gpu > 0:
            template['resources']['limits'] = {'nvidia.com/gpu': self._cluster_config.worker.gpu}
        else:
            if name == 'worker':
                template['resources']['requests'] = {
                    'cpu': self._cluster_config.worker.cpu / 2.0 + 0.1
                }

        return template

    def make_deployment(self, name, replicas):
        template = {
            'apiVersion': 'apps/v1beta1',
            'kind': 'Deployment',
            'metadata': {'name': 'scanner-{}'.format(name)},
            'spec': {  # DeploymentSpec
                'replicas': replicas,
                'template': {
                    'metadata': {'labels': {'app': 'scanner-{}'.format(name)}},
                    'spec': {  # PodSpec
                        'containers': [self.make_container(name)],
                        'volumes': [{
                            'name': 'service-key',
                            'secret': {
                                'secretName': 'service-key',
                                'items': [{
                                    'key': 'service-key.json',
                                    'path': 'service-key.json'
                                }]
                            }
                        }],
                        'nodeSelector': {
                            'cloud.google.com/gke-nodepool':
                            'default-pool' if name == 'master' else 'workers'
                        }
                    }
                }
            }
        }  # yapf: disable

        return template

    def kube_start(self, reset=True):
        self.get_credentials()

        cfg = self._cluster_config
        deploy = self.get_object(self.get_kube_info('deployments'), 'scanner-worker')
        if deploy is not None and cfg.num_workers == 1:
            num_workers = deploy['status']['replicas']
        else:
            num_workers = cfg.num_workers

        if reset:
            run('kubectl delete service --all')
            run('kubectl delete deploy --all')

        secrets = self.get_kube_info('secrets')
        print('Making secrets...')
        if self.get_object(secrets, 'service-key') is None:
            run('kubectl create secret generic service-key --from-file={}' \
                .format(self._cloud_config.service_key))

        if self.get_object(secrets, 'aws-storage-key') is None:
            run('kubectl create secret generic aws-storage-key --from-literal=AWS_ACCESS_KEY_ID={} --from-literal=AWS_SECRET_ACCESS_KEY={}' \
                .format(self._cloud_config.storage_key_id, self._cloud_config.storage_key_secret))

        deployments = self.get_kube_info('deployments')
        print('Creating deployments...')
        if self.get_object(deployments, 'scanner-master') is None:
            self.create_object(self.make_deployment('master', 1))

        services = self.get_kube_info('services')
        if self.get_object(services, 'scanner-master') is None:
            run('kubectl expose deploy/scanner-master --type=NodePort --port=8080')

        if self.get_object(deployments, 'scanner-worker') is None:
            self.create_object(self.make_deployment('worker', num_workers))

    def resize(self, size):
        if not self._cluster_config.autoscale:
            run('{cmd} resize {id} -q --node-pool=workers --size={size}' \
                .format(cmd=self._cluster_cmd, id=self._cluster_config.id, size=size))
        else:
            run('{cmd} update {id} -q --node-pool=workers --enable-autoscaling --max-nodes={size}' \
                .format(cmd=self._cluster_cmd, id=self._cluster_config.id, size=size))

        run('kubectl scale deploy/scanner-worker --replicas={}'.format(size))

    def delete(self):
        run('{cmd} delete {id}'.format(cmd=self._cluster_cmd, id=self._cluster_config.id))

    def cli(self):
        parser = argparse.ArgumentParser()
        command = parser.add_subparsers(dest='command')
        create = command.add_parser('start')
        create.add_argument('--reset', '-r', action='store_true', help='Delete current deployments')
        create.add_argument('--num-workers', '-n', type=int, default=1, help='Initial number of workers')
        command.add_parser('delete')
        resize = command.add_parser('resize')
        resize.add_argument('size', type=int, help='Number of nodes')

        args = parser.parse_args()
        if args.command == 'start':
            if not self.running():
                self.machine_start()
            self.kube_start()

        elif args.command == 'delete':
            self.delete()

        elif args.command == 'resize':
            self.resize(args.size)
