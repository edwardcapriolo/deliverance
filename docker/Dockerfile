ARG JDK_DEVEL_TAG=0.0.6-devel
ARG JDK_RUNTIME_TAG=0.0.6-devel
ARG JDK_IMAGE_REPO=ecapriolo/jdk-25

FROM ${JDK_IMAGE_REPO}:${JDK_DEVEL_TAG} AS reckoning-build
ARG REPO_URL=https://github.com/edwardcapriolo/reckoning.git
ARG REPO_COMMIT_SHA=main
ARG MAVEN_OPTS="-XX:TieredStopAtLevel=1"
ENV MAVEN_OPTS=${MAVEN_OPTS}

RUN apk add --no-cache git maven curl jq unzip bash make clang gcc g++ libc-dev linux-headers
WORKDIR /build
RUN git clone "$REPO_URL" reckoning
WORKDIR /build/reckoning
RUN git checkout "$REPO_COMMIT_SHA"
RUN --mount=type=cache,target=/root/.m2 mvn dependency:go-offline -Dmaven.javadoc.skip=true -Dgpg.skip=true
RUN --mount=type=cache,target=/root/.m2 mvn install -Dmaven.test.skip.exec=true -Dmaven.javadoc.skip=true -Dgpg.skip=true -Dreckoning.bundle.skip=false

FROM ${JDK_IMAGE_REPO}:${JDK_RUNTIME_TAG} AS reckoning
RUN apk add --no-cache bash libstdc++ \
 && mkdir /reckoning \
 && addgroup -S reckoning \
 && adduser -S -G reckoning -H -D reckoning \
 && mkdir /reckoning/logs \
 && chown reckoning:reckoning /reckoning/logs

COPY --from=reckoning-build /build/reckoning/web/target/web-*-exec.jar /reckoning/web.jar
COPY entry_point.sh /reckoning/entry_point.sh
COPY inlinerules.json /reckoning/inlinerules.json
COPY simple.yaml /reckoning/

RUN chmod 755 /reckoning/entry_point.sh
WORKDIR /reckoning
USER reckoning
ENTRYPOINT ["/reckoning/entry_point.sh"]
