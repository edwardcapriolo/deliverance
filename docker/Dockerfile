ARG JDK_DEVEL_TAG=0.0.6-devel
ARG JDK_RUNTIME_TAG=0.0.6-devel
ARG JDK_IMAGE_REPO=ecapriolo/jdk-25

FROM ${JDK_IMAGE_REPO}:${JDK_DEVEL_TAG} AS deliverance-build
ARG REPO_URL=https://github.com/edwardcapriolo/deliverance.git
ARG REPO_COMMIT_SHA=main
ARG MAVEN_OPTS="-XX:TieredStopAtLevel=1"
ENV MAVEN_OPTS=${MAVEN_OPTS}

RUN apk add --no-cache git maven curl jq unzip bash make clang gcc g++ libc-dev linux-headers
WORKDIR /build
RUN git clone "$REPO_URL" deliverance
WORKDIR /build/deliverance
RUN git checkout "$REPO_COMMIT_SHA"
RUN --mount=type=cache,target=/root/.m2 mvn dependency:go-offline -Dmaven.javadoc.skip=true -Dgpg.skip=true
RUN --mount=type=cache,target=/root/.m2 mvn install -Dmaven.test.skip.exec=true -Dmaven.javadoc.skip=true -Dgpg.skip=true

FROM ${JDK_IMAGE_REPO}:${JDK_RUNTIME_TAG} AS deliverance
RUN apk add --no-cache bash libstdc++ \
 && mkdir /deliverance \
 && addgroup -S deliverance \
 && adduser -S -G deliverance -H -D deliverance \
 && mkdir /deliverance/logs \
 && chown deliverance:deliverance /deliverance/logs

COPY --from=deliverance-build /build/deliverance/web/target/web-*.jar /deliverance/web.jar
COPY entry_point.sh /deliverance/entry_point.sh
COPY inlinerules.json /deliverance/inlinerules.json
COPY simple.properties /deliverance/

RUN chmod 755 /deliverance/entry_point.sh
WORKDIR /deliverance
USER deliverance
ENTRYPOINT ["/deliverance/entry_point.sh"]
