# Deliverance Tensor Parallel Helm Chart

This chart starts a Gemma2 tensor-parallel deployment using the current UDP gossip transport.

Default topology:

* 4 worker pods in a StatefulSet.
* 1 coordinator/web pod in a Deployment.
* 4 tensor-parallel ranks total.
* 1 rank per worker pod.
* `tjake/gemma-2-2b-it-JQ4`.

## Image Requirements

The image must contain:

* `/deliverance/web.jar` for the web coordinator.
* A classpath at `/deliverance/lib/*:/deliverance/web.jar` that can run
  `io.teknek.deliverance.benchmark.TpLocalCluster` for workers.
* JDK 25.
* Native libraries if using SIMD/native paths.

The current repo Dockerfile may need adjustment before this chart can run workers unchanged.
Override `workers.command`, `workers.baseArgs`, `coordinator.command`, and `coordinator.args` if your image uses a different layout.

## Install

```sh
helm install deliverance-tp charts/deliverance-tp \
  --set image.repository=your-registry/deliverance \
  --set image.tag=your-tag
```

Port-forward the coordinator:

```sh
kubectl port-forward svc/deliverance-tp-deliverance-tp-coordinator 8080:8080
```

Then use the normal Deliverance web API.

## Notes

* Kubernetes supports UDP Services; this chart uses UDP gossip on port `42606`.
* The worker StatefulSet uses stable DNS names as gossip seeds.
* Rank and collective HTTP endpoints are currently dynamically allocated and published through gossip.
* If pod-to-pod direct HTTP to published pod host/port does not work in your cluster, Deliverance will need advertised host/port support for rank and collective servers.
* HTTP gossip exists in newer local Gossip source but is not available in the current released Gossip dependency used by Deliverance.
