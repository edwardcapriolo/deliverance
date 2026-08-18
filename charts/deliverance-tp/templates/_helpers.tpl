{{- define "deliverance-tp.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "deliverance-tp.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name (include "deliverance-tp.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "deliverance-tp.workerName" -}}
{{- printf "%s-worker" (include "deliverance-tp.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "deliverance-tp.coordinatorName" -}}
{{- printf "%s-coordinator" (include "deliverance-tp.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "deliverance-tp.workerSeedArgs" -}}
{{- $workerName := include "deliverance-tp.workerName" . -}}
{{- range $i := until (int .Values.workers.replicas) }}
- --seed
- worker-{{ $i }}=udp://{{ $workerName }}-{{ $i }}.{{ $workerName }}.$(POD_NAMESPACE).svc.cluster.local:{{ $.Values.tensorParallel.gossipPort }}
{{- end -}}
{{- end -}}

{{- define "deliverance-tp.workerCommonArgs" -}}
- --cluster
- {{ .Values.tensorParallel.cluster | quote }}
- --deployment
- {{ .Values.tensorParallel.deployment | quote }}
- --tensor-parallel-size
- {{ .Values.tensorParallel.size | quote }}
- --max-ranks-per-worker
- {{ .Values.tensorParallel.maxRanksPerWorker | quote }}
- --collective-transport
- {{ .Values.tensorParallel.collectiveTransport | quote }}
- --owner
- {{ .Values.model.owner | quote }}
- --model
- {{ .Values.model.name | quote }}
- --pool-size
- {{ .Values.workers.poolSize | quote }}
- --working-dtype
- {{ .Values.workers.workingDType | quote }}
- --working-qtype
- {{ .Values.workers.workingQType | quote }}
- --output-head-quantization
- {{ .Values.model.outputHeadQuantization | quote }}
{{- if .Values.workers.profileStages }}
- --profile-stages
{{- else }}
- --no-profile-stages
{{- end }}
{{- end -}}
