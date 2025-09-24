package io.teknek.deliverance.fetch;


import com.google.common.io.CountingInputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.zip.GZIPInputStream;

public class HttpSupport {
    public static final Logger logger = LoggerFactory.getLogger(HttpSupport.class);

    public static Pair<InputStream, Long> getResponse(
            String urlString,
            Optional<String> optionalAuthHeader,
            Optional<Pair<Long, Long>> optionalByteRange
    ) throws IOException {
        URL url = new URL(urlString);
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        connection.setRequestMethod("GET");
        connection.setRequestProperty("Accept-Encoding", "gzip");
        optionalAuthHeader.ifPresent(authHeader -> connection.setRequestProperty("Authorization", "Bearer " + authHeader));
        optionalByteRange.ifPresent(byteRange -> connection.setRequestProperty("Range", "bytes="
                + byteRange.getLeft() + "-" + byteRange.getRight()));
        int responseCode = connection.getResponseCode();

        if (responseCode == HttpURLConnection.HTTP_OK || responseCode == HttpURLConnection.HTTP_PARTIAL) {
            String encoding = connection.getContentEncoding();
            InputStream inputStream;
            if (encoding != null && encoding.equals("gzip")) {
                inputStream = new GZIPInputStream(connection.getInputStream());
            } else {
                inputStream = connection.getInputStream();
            }
            return Pair.of(inputStream, connection.getContentLengthLong());
        } else {
            throw new IOException("HTTP response code: " + responseCode + " for URL: " + urlString);
        }
    }

    public static String readInputStream(InputStream inStream) throws IOException {
        if (inStream == null) {
            return null;
        }
        BufferedReader inReader = new BufferedReader(new InputStreamReader(inStream));
        StringBuilder stringBuilder = new StringBuilder();
        String currLine;
        while ((currLine = inReader.readLine()) != null) {
            stringBuilder.append(currLine);
            stringBuilder.append(System.lineSeparator());
        }
        return stringBuilder.toString();
    }

    public static void downloadFile(
            String hfModel,
            String currFile,
            Optional<String> optionalBranch,
            Optional<String> optionalAuthHeader,
            Optional<Pair<Long, Long>> optionalByteRange,
            Path outputPath
    ) throws IOException {

        Pair<InputStream, Long> stream = getResponse(
                "https://huggingface.co/" + hfModel + "/resolve/" + optionalBranch.orElse("main") + "/" + currFile,
                optionalAuthHeader,
                optionalByteRange
        );

        CountingInputStream inStream = new CountingInputStream(stream.getLeft());
        long totalBytes = stream.getRight();
        if (outputPath.toFile().exists() && outputPath.toFile().length() == totalBytes) {
            logger.debug("File already exists: {}", outputPath);
            return;
        }
        CompletableFuture<Long> result = CompletableFuture.supplyAsync(() -> {
            try {
                return Files.copy(inStream, outputPath, StandardCopyOption.REPLACE_EXISTING);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });

        try {
            result.get();
        } catch (Throwable e) {
            throw new IOException("Failed to download file: " + currFile, e);
        }

        if (!result.isCompletedExceptionally()) {
            logger.info("Downloaded file: {}", outputPath);
        }
    }
}
