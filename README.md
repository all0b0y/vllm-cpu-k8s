# vLLM на CPU в Kubernetes (minikube)

Развёртывание [vLLM](https://github.com/vllm-project/vllm) 0.31.0 **строго на CPU** в локальном кластере minikube
с упором на безопасность, плюс замеры TTFT / Latency / Throughput на двух машинах:
Apple M3 (macOS, arm64) и Intel i5-13400F (Windows 11 + WSL2, x86_64).

> Статус: развёртывание и проверки безопасности готовы на Mac; бенчмарки и прогон на Windows — в работе.

## Что внутри

```
клиент (role=client) ──► Service vllm:8000 ──► nginx (8080) ──► guard-middleware ──► vLLM (127.0.0.1:8000)
                                                   │                   │
                                         allowlist путей,      Bearer-токен для всего, кроме /health;
                                         лимит тела 8 MiB,     лимиты max_tokens / n / размер тела;
                                         лимит соединений      одна строка лога на запрос без текста
initContainer model-fetch ──► Hugging Face (только при первом старте) ──► PVC models (sha256 при каждом старте)
```

| Требование | Как сделано |
|---|---|
| Свой Dockerfile, зафиксированные зависимости | `docker/vllm/Dockerfile`: `ubuntu:24.04` по digest, официальный CPU-wheel vLLM, `requirements.txt` со sha256 для обеих архитектур (`uv pip compile --universal --generate-hashes`), apt-снапшот на amd64 |
| Не root | UID 10001, `readOnlyRootFilesystem`, `drop: ALL`, seccomp `RuntimeDefault`, namespace с Pod Security `restricted` |
| Логи без промптов и ответов | `src/vllm_guard/logfilter.py` + `docker/vllm/logging.json`; entrypoint не стартует с флагами логирования контента; проверяется canary-тестом |
| PVC-кэш модели | `models` (10 Gi); повторный старт идёт без сети, только проверка sha256 |
| Secret для HF-токена и API-ключа | kustomize `secretGenerator` из gitignored-файлов (`scripts/make-secrets.sh`); HF-токен смонтирован файлом только в initContainer; API-ключ — env из Secret, не флаг CLI |
| ConfigMap | `vllm-config` (YAML для `vllm serve --config`) и `vllm-env`; хеш в имени → под перезапускается при изменении |
| Бонус: initContainer-загрузчик | `src/loader/fetch_model.py` в distroless-образе: allowlist моделей + ревизия-коммит, сканеры HF, только safetensors, двойная сверка sha256 (метаданные HF и `models.lock.json`) |
| Бонус: TTFT / Latency / Throughput | см. «Результаты» (в работе) |

## Быстрый старт (Mac)

Нужны Docker Desktop (8 CPU, ≥ 11 GB RAM), minikube ≥ 1.39, kubectl.

```bash
make build                                   # образы vllm-cpu-guard:0.31.0 и model-loader:1
bash scripts/make-secrets.sh --no-hf         # API-ключ; для закрытых моделей — без --no-hf (спросит токен)
scripts/minikube-up.sh                       # minikube + Calico + CPU Manager static
minikube image load vllm-cpu-guard:0.31.0 && minikube image load model-loader:1
kubectl apply -k k8s/overlays/mac-m3         # первый старт: загрузка модели (~5 мин)
kubectl -n llm rollout status deploy/vllm --timeout=20m
```

Windows (WSL2, Ubuntu 24.04): те же команды в bash внутри WSL; `CPUS=16 RESERVED=0-1 scripts/minikube-up.sh`
и overlay `k8s/overlays/win-i5`.

## Проверки

```bash
make test                                    # юнит-тесты: guard, фильтр логов, entrypoint, загрузчик
scripts/k8s-test.sh smoke                    # из пода role=client: 401 без токена, лимиты, ответ, стрим
scripts/k8s-test.sh logs                     # canary, ответ модели и ключ не встречаются в логах пода
scripts/k8s-test.sh attacker                 # под без role=client не достучится (NetworkPolicy)
scripts/k8s-test.sh offline                  # перезапуск без какого-либо egress: модель берётся из PVC
HF_REPO=user/repo HF_REV=<sha> scripts/hf-private-test.sh   # приватный репозиторий: без токена / неверный / верный
make scan                                    # hadolint + Trivy (образы и манифесты)
```

Для сравнения можно поднять тот же образ «как есть» (`MODE=stock scripts/docker-smoke.sh`): у штатного
`vllm serve --api-key` без токена открыты `/invocations` (генерация!), `/metrics`, `/openapi.json`, `/load` и др.

## Результаты

Заполняется по итогам бенчмарков: команда запуска теста, таблица TTFT / Latency / Throughput для обеих машин,
аналитика случаев А/Б, затем графики загрузки CPU / RAM по фазам.
