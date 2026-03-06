#!/bin/bash
# Script to run the LEAF model example

# Set JAVA_HOME if not already set
if [ -z "$JAVA_HOME" ]; then
    if [ -x /usr/libexec/java_home ]; then
        # macOS
        export JAVA_HOME="$(/usr/libexec/java_home)"
    elif command -v java &> /dev/null; then
        # Linux / other: resolve from java on PATH
        export JAVA_HOME="$(java -XshowSettings:property -version 2>&1 | grep 'java.home' | awk '{print $3}')"
    fi
    if [ -z "$JAVA_HOME" ]; then
        echo "ERROR: JAVA_HOME is not set and could not be detected. Install Java or set JAVA_HOME."
        exit 1
    fi
fi

cd "$(dirname "$0")" || exit 1

# Build and install core and its sibling dependencies if needed
CORE_JAR_EXISTS=$(ls core/target/core-*.jar 2>/dev/null | grep -v sources | grep -v javadoc | head -1)
if [ -z "$CORE_JAR_EXISTS" ]; then
    echo "Building core and dependencies..."
    mvn clean install -Dmaven.test.skip=true -Dmaven.javadoc.skip=true -Dgpg.skip=true -pl core -am || exit 1
fi

cd core || exit 1

# Copy dependencies if needed
if [ ! -d "target/dependency" ] || [ -z "$(ls -A target/dependency/*.jar 2>/dev/null)" ]; then
    echo "Copying dependencies..."
    CORE_DIR="$(pwd)"
    cd ..
    mvn dependency:copy-dependencies -pl core -DoutputDirectory="$CORE_DIR/target/dependency" || exit 1
    cd "$CORE_DIR" || exit 1
fi

# Build classpath - explicitly add all dependency jars
# Find the core jar (version may vary) - exclude sources and javadoc jars
CORE_JAR=$(ls target/core-*.jar | grep -v sources | grep -v javadoc | head -1)
if [ -z "$CORE_JAR" ]; then
    echo "ERROR: core jar not found in target/"
    exit 1
fi
CP="$CORE_JAR"
DEPENDENCY_COUNT=0
if [ -d "target/dependency" ]; then
    # Add each jar explicitly to avoid wildcard expansion issues
    for jar in target/dependency/*.jar; do
        if [ -f "$jar" ]; then
            CP="$CP:$jar"
            DEPENDENCY_COUNT=$((DEPENDENCY_COUNT + 1))
        fi
    done
fi

if [ "$DEPENDENCY_COUNT" -eq 0 ]; then
    echo "ERROR: No dependency jars found in target/dependency/"
    exit 1
fi

echo "Found $DEPENDENCY_COUNT dependency jars"

# Run the example
echo "Running LEAF model example..."
echo "Classpath: $CP"
echo ""
java -cp "$CP" \
    --add-opens java.base/java.nio=ALL-UNNAMED \
    --add-modules jdk.incubator.vector \
    --enable-preview \
    io.teknek.deliverance.examples.LeafModelExample "$@"
