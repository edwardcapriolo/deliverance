package io.teknek.deliverance.safetensors.fetch;

import org.junit.jupiter.api.Test;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.ByteArrayOutputStream;
import java.net.ServerSocket;
import java.net.URI;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;
import java.util.zip.GZIPOutputStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class HttpSupportTest {

    @Test
    void getResponseSendsAuthAndRangeHeadersAndReadsPartialResponse() throws Exception {
        AtomicReference<String> request = new AtomicReference<>();
        try (TestServer server = new TestServer("HTTP/1.1 206 Partial Content\r\n"
                + "Content-Length: 5\r\n\r\nhello", request)) {
            Pair<InputStream, Long> response = HttpSupport.getResponse(
                    server.uri(), Optional.of("secret"), Optional.of(Pair.of(10L, 14L)));
            try (InputStream body = response.getLeft()) {
                assertEquals("hello" + System.lineSeparator(), HttpSupport.readInputStream(body));
            }
            assertEquals(5L, response.getRight());
        }

        assertTrue(request.get().contains("Authorization: Bearer secret"));
        assertTrue(request.get().contains("Range: bytes=10-14"));
    }

    @Test
    void getResponseRejectsNonSuccessfulStatus() throws Exception {
        AtomicReference<String> request = new AtomicReference<>();
        try (TestServer server = new TestServer("HTTP/1.1 404 Not Found\r\n"
                + "Content-Length: 0\r\n\r\n", request)) {
            assertThrows(IOException.class, () -> HttpSupport.getResponse(
                    server.uri(), Optional.empty(), Optional.empty()));
        }
        assertTrue(request.get().startsWith("GET / HTTP/1.1"));
    }

    @Test
    void getResponseDecompressesGzipBody() throws Exception {
        byte[] compressed;
        try (ByteArrayOutputStream bytes = new ByteArrayOutputStream();
             GZIPOutputStream gzip = new GZIPOutputStream(bytes)) {
            gzip.write("compressed".getBytes(StandardCharsets.UTF_8));
            gzip.finish();
            compressed = bytes.toByteArray();
        }
        byte[] response = ("HTTP/1.1 200 OK\r\n"
                + "Content-Encoding: gzip\r\n"
                + "Content-Length: " + compressed.length + "\r\n\r\n").getBytes(StandardCharsets.ISO_8859_1);
        byte[] fullResponse = new byte[response.length + compressed.length];
        System.arraycopy(response, 0, fullResponse, 0, response.length);
        System.arraycopy(compressed, 0, fullResponse, response.length, compressed.length);

        try (TestServer server = new TestServer(fullResponse, new AtomicReference<>())) {
            Pair<InputStream, Long> result = HttpSupport.getResponse(server.uri(), Optional.empty(), Optional.empty());
            try (InputStream body = result.getLeft()) {
                assertEquals("compressed" + System.lineSeparator(), HttpSupport.readInputStream(body));
            }
        }
    }

    @Test
    void readInputStreamHandlesNullAndMultipleLines() throws Exception {
        assertNull(HttpSupport.readInputStream(null));
        assertEquals("one" + System.lineSeparator() + "two" + System.lineSeparator(),
                HttpSupport.readInputStream(new java.io.ByteArrayInputStream("one\ntwo".getBytes(StandardCharsets.UTF_8))));
    }

    private static final class TestServer implements AutoCloseable {
        private final ServerSocket serverSocket;
        private final byte[] response;
        private final AtomicReference<String> request;
        private final Thread thread;

        private TestServer(String response, AtomicReference<String> request) throws IOException {
            this(response.getBytes(StandardCharsets.ISO_8859_1), request);
        }

        private TestServer(byte[] response, AtomicReference<String> request) throws IOException {
            this.serverSocket = new ServerSocket(0);
            this.response = response;
            this.request = request;
            this.thread = new Thread(this::serve, "http-support-test-server");
            this.thread.start();
        }

        private String uri() {
            return URI.create("http://127.0.0.1:" + serverSocket.getLocalPort() + "/").toString();
        }

        private void serve() {
            try (Socket socket = serverSocket.accept();
                 BufferedReader reader = new BufferedReader(new InputStreamReader(socket.getInputStream(), StandardCharsets.ISO_8859_1));
                 OutputStream output = socket.getOutputStream()) {
                StringBuilder requestText = new StringBuilder();
                String line;
                while (!(line = reader.readLine()).isEmpty()) {
                    requestText.append(line).append('\n');
                }
                request.set(requestText.toString());
                output.write(response);
                output.flush();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        @Override
        public void close() throws Exception {
            serverSocket.close();
            thread.join();
        }
    }
}
