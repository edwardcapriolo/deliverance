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

{{- define "deliverance-tp.cachePvcName" -}}
{{- printf "%s-model-cache" (include "deliverance-tp.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "deliverance-tp.cacheStorageClassName" -}}
{{- default (printf "%s-filestore-rwx" (include "deliverance-tp.fullname" .) | trunc 63 | trimSuffix "-") .Values.cache.storageClassName -}}
{{- end -}}

{{- define "deliverance-tp.workerSeedArgs" -}}
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
- --assignment-mode
- {{ .Values.tensorParallel.assignmentMode | quote }}
- --tensor-operations
- {{ .Values.tensorOperations.type | quote }}
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
- --rank-connect-timeout-seconds
- {{ .Values.tensorParallel.rankConnectTimeoutSeconds | quote }}
- --rank-request-timeout-seconds
- {{ .Values.tensorParallel.rankRequestTimeoutSeconds | quote }}
- --rank-operation-timeout-seconds
- {{ .Values.tensorParallel.rankOperationTimeoutSeconds | quote }}
- --rank-close-timeout-seconds
- {{ .Values.tensorParallel.rankCloseTimeoutSeconds | quote }}
- --admin-port
- {{ .Values.workers.adminPort | quote }}
{{- end -}}
