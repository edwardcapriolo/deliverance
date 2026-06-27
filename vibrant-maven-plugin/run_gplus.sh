export MAVEN_OPTS="--add-opens java.base/java.nio=ALL-UNNAMED --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED --sun-misc-unsafe-memory-access=allow -Dorg.slf4j.simpleLogger.defaultLogLevel=error -Djava.library.path=../native/target/native-lib-only/osx-aarch_64"
mvn gplus:shell
