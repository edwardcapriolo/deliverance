package net.deliverance.http;

import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import org.springframework.http.HttpStatus;
import org.springframework.http.HttpHeaders;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

import java.util.Objects;
import java.util.concurrent.Semaphore;

@RestController
public class FetchController {

    private final Semaphore fetchGate = new Semaphore(1);

    @RequestMapping(method = RequestMethod.POST, value = "/api/fetch", consumes = "application/json", produces = "application/json")
    public ResponseEntity<FetchResponse> fetch(@RequestBody FetchRequest request) {
        if (request == null || isBlank(request.owner()) || isBlank(request.name())) {
            return ResponseEntity.badRequest().body(FetchResponse.failed("owner and name are required"));
        }
        if (!fetchGate.tryAcquire()) {
            return ResponseEntity.status(HttpStatus.TOO_MANY_REQUESTS)
                    .header(HttpHeaders.RETRY_AFTER, "5")
                    .body(FetchResponse.failed("another model is being fetched"));
        }
        try {
            new ModelFetcher(request.owner(), request.name()).maybeDownload();
            return ResponseEntity.ok(FetchResponse.ready(request.owner(), request.name()));
        } catch (RuntimeException e) {
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY)
                    .body(FetchResponse.failed(e.getMessage()));
        } finally {
            fetchGate.release();
        }
    }

    private static boolean isBlank(String value) {
        return value == null || value.isBlank();
    }

    public record FetchRequest(String owner, String name) {
    }

    public enum FetchStatus {
        READY,
        FAILED
    }

    public record FetchResponse(FetchStatus status, String owner, String name, String error) {
        static FetchResponse ready(String owner, String name) {
            return new FetchResponse(FetchStatus.READY, owner, name, null);
        }

        static FetchResponse failed(String error) {
            return new FetchResponse(FetchStatus.FAILED, null, null,
                    Objects.requireNonNullElse(error, "model fetch failed"));
        }
    }
}
