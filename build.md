# Build And Test Guide

Deliverance requires JDK 25 for most modules. The generated OpenAPI `client` module targets Java 17 internally, but normal repo builds should use JDK 25.

## Standard Build

```sh
export JAVA_HOME=/usr/lib/jvm/java-25-temurin-jdk/
git clone git@github.com:edwardcapriolo/deliverance.git
cd deliverance
mvn install -DskipTests
```

Use `-am` when building a module that depends on local modules:

```sh
mvn -q -pl core -am -DskipTests compile
```

## Java-Only Native Build

The `native` module contains Java provider classes and optional native libraries. On systems without a C compiler, `make`, or platform-native toolchain, use the Java-only profile:

```sh
mvn -Pnative-java-only -pl native -am -DskipTests package
```

This compiles the Java classes in `native` but skips:

- Dawn download/extraction
- C/C++ compilation
- `make`
- native smoke-test Ant steps
- native classifier jar assembly steps

Runtime behavior remains fallback-oriented: if native libraries are absent, the Java classes log that native libraries are unavailable and delegate to Panama/Java paths where supported.

For convenience, the repo root has:

```sh
sh build_no_native.sh
```

## Full Native Build

The full native build requires a platform toolchain. On Alpine-like systems, typical packages include:

```sh
doas apk add curl
doas apk add openjdk25
doas apk add gpg
doas apk add bash
doas apk add llvm clang lld
```

The exact compiler package names vary by distribution.

Build native normally with:

```sh
mvn -q -pl native -am -DskipTests package
```

## Native Build Logs

If the native C build fails, Maven may only show an Ant/exec return code. The native module now prints log file locations for full native builds:

- `native/target/native-build.log`
- `native/target/native-smoke-test.log`

Check those files when the console output does not include the compiler or linker error.

## Test Flags

Common Maven flags:

```text
-DskipTests
-DskipITs
-Dmaven.test.skip=true
```

By default, integration tests tagged `large-model` are excluded.

Skip integration tests named `*IT`:

```sh
mvn verify -DskipITs
```

Compile tests but do not execute them:

```sh
mvn install -Dmaven.test.skip.exec=true
```

## Focused Tests

Prefer focused tests while iterating:

```sh
mvn -q -pl tensor -am -Dtest=TensorCopyFromTest test
mvn -q -pl core -am -Dtest=KvBufferCachePrefixTest test
mvn -q -pl native -am -Dtest=NativeGpuGemmParityTest -Dsurefire.failIfNoSpecifiedTests=false test
```

If Maven or javac crashes under the local JDK/JIT, this environment has historically been more stable with:

```sh
MAVEN_OPTS="-XX:TieredStopAtLevel=1" mvn -q -pl core -am -DskipTests compile
```
