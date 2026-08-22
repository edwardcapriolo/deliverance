# GKE Qwen3 0.6B TP8 Runbook

This page describes the GKE deployment shape used to smoke test `edwardcapriolo/Qwen3-0.6B-JQ4` with 8 tensor-parallel ranks.

The model has 8 TP ranks. The chart can place those ranks as:

* 8 workers with 1 rank each, using `values-qwen06b-tp8-gke.yaml`.
* 2 workers with 4 ranks each, using `values-qwen06b-tp8-gke-2x4.yaml` for simpler debugging.

The setup uses one shared ReadWriteMany model cache so the coordinator downloads the model once and workers read the same files from disk. On GKE, the intended shared filesystem is Filestore through the Filestore CSI driver.

## Prerequisites

Enable the Filestore CSI driver:

```sh
gcloud container clusters update int-gke \
  --region us-central1 \
  --update-addons=GcpFilestoreCsiDriver=ENABLED
```

Enable the Cloud Filestore API:

```sh
gcloud services enable file.googleapis.com --project <project-id>
```

Find the cluster VPC network. The Filestore StorageClass must use this network, not necessarily `default`:

```sh
gcloud container clusters describe int-gke \
  --region us-central1 \
  --format='value(network,subnetwork)'
```

Example output:

```text
int-network    int-gke
```

Use the first value as `cache.filestore.network`.

## Install Coordinator First

Start with workers scaled to zero. This lets the coordinator download the model into the shared cache before workers mount and load from it.

```sh
helm upgrade --install model-testing charts/deliverance-tp \
  -n model-testing --create-namespace \
  -f charts/deliverance-tp/values-qwen06b-tp8.yaml \
  -f charts/deliverance-tp/values-qwen06b-tp8-gke.yaml \
  --set workers.replicas=0 \
  --set image.pullPolicy=Always \
  --set cache.filestore.network=int-network \
  --set 'nodeSelector.cloud\.google\.com/gke-nodepool=loadtest-pool' \
  --set 'tolerations[0].key=dedicated' \
  --set 'tolerations[0].operator=Equal' \
  --set 'tolerations[0].value=loadtest' \
  --set 'tolerations[0].effect=NoSchedule'
```

Successful Helm output looks like:

```text
Release "model-testing" does not exist. Installing it now.
NAME: model-testing
LAST DEPLOYED: <date>
NAMESPACE: model-testing
STATUS: deployed
REVISION: 1
DESCRIPTION: Install complete
TEST SUITE: None
```

Watch the shared PVC:

```sh
kubectl get pvc -n model-testing -w
```

Expected shape:

```text
NAME                        STATUS   VOLUME          CAPACITY   ACCESS MODES   STORAGECLASS
model-testing-model-cache   Bound    pvc-<id>        100Gi      RWX            model-testing-filestore-rwx
```

If the PVC stays pending, inspect events:

```sh
kubectl describe pvc -n model-testing model-testing-model-cache
kubectl get events -n model-testing --sort-by='.lastTimestamp'
```

Common issues:

* Filestore CSI driver is not enabled.
* `file.googleapis.com` is disabled.
* StorageClass `network` does not match the cluster VPC.

The wrong-network failure looks like a successful PVC followed by pod mount timeouts. The Filestore node driver logs show
the NFS mount target and timeout:

```text
Mounting command: mount
Mounting arguments: -t nfs <filestore-ip>:/vol1 <kubelet-globalmount-path>
Output: mount.nfs: Connection timed out
```

If that happens, check the StorageClass:

```sh
kubectl get storageclass model-testing-filestore-rwx -o yaml
```

The important field is:

```yaml
parameters:
  network: int-network
```

If the network is wrong, delete the pod, PVC, and StorageClass, then rerun Helm with the corrected
`cache.filestore.network` value. Deleting the PVC deletes the backing Filestore instance and can take several minutes.

Watch coordinator logs until the model is downloaded and Spring starts:

```sh
kubectl logs -n model-testing deploy/model-testing-coordinator -f
```

An anonymized successful startup includes:

```text
Downloaded file: /home/deliverance/.deliverance/edwardcapriolo_Qwen3-0.6B-JQ4/model.safetensors
Tensor provider = Native SIMD Operations
Spring tensor-parallel coordinator model loaded model=Qwen3-0.6B-JQ4 deployment=model-testing size=8 initialState=NOT_READY
Tomcat started on port 8080 (http)
```

A fuller successful coordinator-first log sequence looks like:

```text
Starting tensor-parallel gossip membership cluster=model-testing node=10.x.x.10 uri=udp://10.x.x.10:42606 deployment=model-testing requestedRanks=8 maxRanksPerNode=1
Started tensor-parallel gossip membership cluster=model-testing node=10.x.x.10 uri=udp://10.x.x.10:42606
Downloaded file: /home/deliverance/.deliverance/edwardcapriolo_Qwen3-0.6B-JQ4/README.md
Downloaded file: /home/deliverance/.deliverance/edwardcapriolo_Qwen3-0.6B-JQ4/config.json
Downloaded file: /home/deliverance/.deliverance/edwardcapriolo_Qwen3-0.6B-JQ4/model.safetensors
Downloaded file: /home/deliverance/.deliverance/edwardcapriolo_Qwen3-0.6B-JQ4/tokenizer.json
Downloaded file: /home/deliverance/.deliverance/edwardcapriolo_Qwen3-0.6B-JQ4/tokenizer_config.json
Tensor provider = Native SIMD Operations, parallelSplitSize = 2
Model type = Q4, Working memory type = F32, Quantized memory type = I8
Spring tensor-parallel coordinator model loaded model=Qwen3-0.6B-JQ4 deployment=model-testing size=8 initialState=NOT_READY
Tomcat started on port 8080 (http) with context path '/'
Started DeliveranceApplication in 32.125 seconds
```

## Scale Workers

For the 8-worker shape:

```sh
kubectl scale -n model-testing statefulset/model-testing-worker --replicas=8
```

For the 2x4 debugging shape, install with `values-qwen06b-tp8-gke-2x4.yaml` and scale to 2 workers.

Watch pods:

```sh
kubectl get pods -n model-testing -o wide -w
```

Example anonymized shape:

```text
NAME                         READY   STATUS    RESTARTS   AGE   IP
model-testing-coordinator    1/1     Running   0          14m   10.x.a.10
model-testing-worker-0       1/1     Running   0          34s   10.x.b.11
model-testing-worker-1       1/1     Running   0          38s   10.x.a.12
model-testing-worker-2       1/1     Running   0          62s   10.x.c.13
model-testing-worker-3       1/1     Running   0          55s   10.x.b.14
model-testing-worker-4       1/1     Running   0          53s   10.x.a.15
model-testing-worker-5       1/1     Running   0          48s   10.x.c.16
model-testing-worker-6       1/1     Running   0          46s   10.x.b.17
model-testing-worker-7       1/1     Running   0          43s   10.x.a.18
```

Worker logs should show that each worker loads from the shared cache and starts a rank server. For one-rank-per-worker:

```sh
kubectl logs -n model-testing pod/model-testing-worker-0 --tail=120
```

Example:

```text
Starting TP local process role=WORKER cluster=model-testing node=10.x.b.11 uri=udp://10.x.b.11:42606 owner=edwardcapriolo model=Qwen3-0.6B-JQ4 deployment=model-testing tpSize=8 maxRanksPerWorker=1
Tensor provider = Native SIMD Operations, parallelSplitSize = 16
Model type = Q4, Working memory type = F32, Quantized memory type = I8
Published tensor-parallel capacity node=10.x.b.11 deployment=model-testing slots=1
Started HTTP tensor-parallel rank server uri=http://10.x.b.11:<port>
Started tensor-parallel rank server node=10.x.b.11 rank=0 size=8 uri=http://10.x.b.11:<port> provider=Native SIMD Operations
Published tensor-parallel rank endpoints node=10.x.b.11 deployment=model-testing endpoints=[TensorParallelRankEndpoint[rank=0, nodeId=10.x.b.11, uri=http://10.x.b.11:<port>]]
```

## Check Readiness

Port-forward the coordinator:

```sh
kubectl port-forward -n model-testing svc/model-testing-coordinator 8080:8080
```

Check gossip:

```sh
curl -s http://127.0.0.1:8080/tp/gossip | jq
```

Successful output should show all worker IDs in `liveMembers` and `candidates`, plus a complete assignment:

```json
{
  "Qwen3-0.6B-JQ4": {
    "nodeId": "10.x.x.10",
    "liveMembers": ["10.x.x.11", "10.x.x.12", "..."],
    "candidates": ["10.x.x.11", "10.x.x.12", "..."],
    "leader": "10.x.x.11",
    "assignment": "TensorParallelAssignment[...]"
  },
  "tensorParallelModels": 1
}
```

A fuller 8-rank example looks like:

```json
{
  "Qwen3-0.6B-JQ4": {
    "nodeId": "10.x.a.10",
    "liveMembers": [
      "10.x.a.12",
      "10.x.a.15",
      "10.x.a.18",
      "10.x.b.11",
      "10.x.b.14",
      "10.x.b.17",
      "10.x.c.13",
      "10.x.c.16"
    ],
    "candidates": [
      "10.x.a.12",
      "10.x.a.15",
      "10.x.a.18",
      "10.x.b.11",
      "10.x.b.14",
      "10.x.b.17",
      "10.x.c.13",
      "10.x.c.16"
    ],
    "leader": "10.x.a.12",
    "assignment": "TensorParallelAssignment[deploymentId=model-testing, leaderNodeId=10.x.a.12, tensorParallelSize=8, ranks=[rank 0..7 assigned to the eight worker IPs]]"
  },
  "tensorParallelModels": 1
}
```

Check endpoints:

```sh
curl -s http://127.0.0.1:8080/tp/endpoints | jq
```

Expected shape:

```json
{
  "Qwen3-0.6B-JQ4": {
    "nodeId": "10.x.a.10",
    "collectiveUri": "netty://10.x.a.12:<port>",
    "rankEndpoints": [
      {"rank": 0, "nodeId": "10.x.c.13", "uri": "http://10.x.c.13:<port>"},
      {"rank": 1, "nodeId": "10.x.c.16", "uri": "http://10.x.c.16:<port>"},
      {"rank": 2, "nodeId": "10.x.b.14", "uri": "http://10.x.b.14:<port>"},
      {"rank": 3, "nodeId": "10.x.b.17", "uri": "http://10.x.b.17:<port>"},
      {"rank": 4, "nodeId": "10.x.b.11", "uri": "http://10.x.b.11:<port>"},
      {"rank": 5, "nodeId": "10.x.a.15", "uri": "http://10.x.a.15:<port>"},
      {"rank": 6, "nodeId": "10.x.a.18", "uri": "http://10.x.a.18:<port>"},
      {"rank": 7, "nodeId": "10.x.a.12", "uri": "http://10.x.a.12:<port>"}
    ]
  },
  "tensorParallelModels": 1
}
```

Check coordinator readiness:

```sh
curl -s http://127.0.0.1:8080/tp/status | jq '."Qwen3-0.6B-JQ4" | {state, diagnostics, groupOpen}'
```

Expected:

```json
{
  "state": "READY",
  "diagnostics": "ready",
  "groupOpen": true
}
```

Coordinator logs when readiness opens the generation group look like:

```text
Opening tensor-parallel generation group node=10.x.a.10 deployment=model-testing tensorParallelSize=8 endpoints=[TensorParallelRankEndpoint[rank=0, ...], ..., TensorParallelRankEndpoint[rank=7, ...]]
Reactivated tensor-parallel generation group assignmentHash=<hash> endpoints=[TensorParallelRankEndpoint[rank=0, ...], ..., TensorParallelRankEndpoint[rank=7, ...]]
```

## Run A Query

```sh
curl -s http://127.0.0.1:8080/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-0.6B-JQ4","messages":[{"role":"user","content":"Say hello in one short sentence."}],"max_tokens":16,"temperature":0}' | jq
```

An anonymized successful response shape:

```json
{
  "choices": [
    {
      "finish_reason": "length",
      "message": {
        "role": "assistant",
        "content": "Okay, the user wants me to write a short, one-sentence"
      }
    }
  ]
}
```

Coordinator logs include readiness, prefix-cache, and timing lines:

```text
Tensor-parallel request readiness OK assignmentHash=<hash> endpoints=[TensorParallelRankEndpoint[rank=0, ...], ..., TensorParallelRankEndpoint[rank=7, ...]]
TP prefix probe miss reason=rank_miss rank=0
time_to_first_token=1509.633435 prefix_length=0
generation_complete prompt_tokens=16 generated_tokens=16 total_ms=12701.021 ttft_ms=1509.633 tokens_per_second=1.260 decode_tokens_per_second=1.340 finish_reason=MAX_TOKENS
```

The timing numbers above are an example from a small GKE node pool, not a performance target. For performance tests, use
larger dedicated nodes, set CPU/memory requests, and avoid oversubscribing worker pools.

## Topology Changes

When changing rank placement, for example from 2 workers x 4 ranks to 8 workers x 1 rank, restart the in-memory gossip
cluster without deleting the shared cache:

```sh
kubectl scale -n model-testing deploy/model-testing-coordinator --replicas=0
kubectl scale -n model-testing statefulset/model-testing-worker --replicas=0
kubectl wait -n model-testing --for=delete pod -l app.kubernetes.io/name=model-testing --timeout=180s
kubectl scale -n model-testing deploy/model-testing-coordinator --replicas=1
kubectl scale -n model-testing statefulset/model-testing-worker --replicas=8
```

After the restart, run the readiness checks again. Do not delete `model-testing-model-cache` unless you want to destroy the
model cache and the backing Filestore instance.

## Operational Notes

* Do not delete the shared PVC unless you want to delete the shared model cache and backing Filestore instance.
* For topology changes, such as 2 workers x 4 ranks to 8 workers x 1 rank, scale coordinator and workers to zero and then scale back up.
* Set `image.pullPolicy=Always` when reusing a mutable snapshot tag.
* Filestore provisioning and deletion can take several minutes.
* Filestore is billed by provisioned capacity and tier, not just used bytes.

Scale down without deleting the cache:

```sh
kubectl scale -n model-testing statefulset/model-testing-worker --replicas=0
kubectl scale -n model-testing deploy/model-testing-coordinator --replicas=0
```
