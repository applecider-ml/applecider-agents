FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*
WORKDIR /opt
COPY . /opt/applecider_agents
# flare (models included) from its public repository; the benchmark tables are fetched on first use
RUN pip install --no-cache-dir "git+https://github.com/applecider-ml/flare.git#egg=flare-transients[skyportal]" \
 && pip install --no-cache-dir "/opt/applecider_agents[mcp]"
ENV APPLECIDER_AGENTS_HOME=/var/lib/applecider-agents APPLECIDER_AGENTS_LOG=/var/log/applecider-agents FLARE_MODE=dry-run
VOLUME ["/var/lib/applecider-agents", "/var/log/applecider-agents"]
ENTRYPOINT ["applecider-agents"]
CMD ["nightly", "--since-days", "1"]
