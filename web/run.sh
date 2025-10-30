export JAVA_HOME=/usr/lib/jvm/java-24-temurin-jdk
PATH=$JAVA_HOME/bin:$PATH
#mvn spring-boot:run -Dspring-boot.run.jvmArguments="--add-modules jdk.incubator.vector -Xmx6G -Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0 --add-opens=jdk.incubator.vector/jdk.incubator.vector=ALL-UNNAMED --add-modules=jdk.incubator.vector --add-exports java.base/sun.nio.ch=ALL-UNNAMED --enable-preview --enable-native-access=ALL-UNNAMED \
# -XX:+UnlockDiagnosticVMOptions -XX:CompilerDirectivesFile=../inlinerules.json -XX:+AlignVector -XX:+UseStringDeduplication \
# -XX:+UseCompressedOops -XX:+UseCompressedClassPointers"
#java --add-modules jdk.incubator.vector -Xmx6G -cp target/web-0.0.1-SNAPSHOT.jar com.deliverance.http.DeliveranceApplication
java --add-modules jdk.incubator.vector -Xmx6G -jar target/web-0.0.1-SNAPSHOT.jar
