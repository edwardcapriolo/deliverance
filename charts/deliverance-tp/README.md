# Deliverance Tensor Parallel Helm Chart

This chart starts a Deliverance tensor-parallel deployment using the current gossip transport.

Default topology:

* 4 worker pods in a StatefulSet.
* 1 coordinator/web pod in a Deployment.
* 4 tensor-parallel ranks total.
* 1 rank per worker pod.
* `tjake/gemma-2-2b-it-JQ4` by default, with Qwen values available in `values-qwen06b-tp8.yaml`.
* One shared ReadWriteMany model cache PVC mounted by the coordinator and workers.

## Image Requirements

The image must contain:

* `/deliverance/web.jar` for the web coordinator.
* A Spring Boot executable jar layout containing `BOOT-INF/classes` and `BOOT-INF/lib`; workers extract that
  classpath at startup to run `io.teknek.deliverance.benchmark.TpLocalCluster`.
* JDK 25.
* Native libraries if using SIMD/native paths.

Override `workers.command`, `workers.baseArgs`, `coordinator.command`, and `coordinator.args` if your image uses a different layout.

## Install

```sh
helm install deliverance-tp charts/deliverance-tp \
  --set image.repository=ecapriolo/deliverance \
  --set image.tag=0.0.14-SNAPSHOT
```

Install Qwen3 0.6B TP8 values:

```sh
helm upgrade --install deliverance-qwen charts/deliverance-tp \
  --namespace deliverance --create-namespace \
  -f charts/deliverance-tp/values-qwen06b-tp8.yaml
```

For GKE, add the GKE overlay. It pins pods to amd64 nodes, uses a shared Filestore CSI RWX cache, and names installed
resources around `model-testing` so test resources are easy to identify:

```sh
helm upgrade --install model-testing charts/deliverance-tp \
  --namespace model-testing --create-namespace \
  -f charts/deliverance-tp/values-qwen06b-tp8.yaml \
  -f charts/deliverance-tp/values-qwen06b-tp8-gke.yaml
```

Port-forward the coordinator:

```sh
kubectl port-forward svc/deliverance-tp-deliverance-tp-coordinator 8080:8080
```

Then use the normal Deliverance web API.

See [GKE_QWEN_TP8.md](GKE_QWEN_TP8.md) for the full coordinator-first install, shared Filestore cache setup, scale-up,
diagnostics, and query runbook.

## Notes

* Kubernetes supports UDP Services; this chart uses UDP gossip on port `42606`.
* Rank HTTP endpoints and collective endpoints are dynamically allocated and published through gossip.
* The shared cache uses GKE Filestore CSI. The Filestore CSI driver and `file.googleapis.com` must be enabled, and the StorageClass network must match the cluster VPC.
