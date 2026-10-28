package io.teknek.deliverance.inferj;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.teknek.deliverance.client.api.ModelsApi;
import io.teknek.deliverance.client.core.ApiClient;
import io.teknek.deliverance.client.model.FetchRequest;
import io.teknek.deliverance.client.model.FetchResponse;
import io.teknek.deliverance.client.model.ServeRequest;
import io.teknek.deliverance.client.model.ServeResponse;
import okhttp3.OkHttpClient;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.jackson.JacksonConverterFactory;
import retrofit2.converter.scalars.ScalarsConverterFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

public final class InferjCli {
    private static final ObjectMapper JSON = new ObjectMapper();

    private InferjCli() {
    }

    public static void main(String[] args) throws Exception {
        if (args.length == 0 || "--help".equals(args[0])
                || (args.length > 1 && "--help".equals(args[1]))) {
            usage();
            return;
        }

        String command = args[0];
        Options options = Options.parse(args);
        ModelsApi api = modelsApi(options.endpoint());
        switch (command) {
            case "fetch" -> fetch(api, options);
            case "serve" -> serve(api, options);
            default -> throw new IllegalArgumentException("Unknown command: " + command);
        }
    }

    private static void fetch(ModelsApi api, Options options) throws IOException {
        Response<FetchResponse> response = api.fetchModel(new FetchRequest()
                .owner(options.owner())
                .name(options.name())).execute();
        requireSuccess(response, "fetch");
        FetchResponse body = response.body();
        System.out.println("Fetched " + body.getOwner() + "/" + body.getName());
    }

    private static void serve(ModelsApi api, Options options) throws IOException {
        ServeRequest request = new ServeRequest()
                .owner(options.owner())
                .name(options.name())
                .type(ServeRequest.TypeEnum.valueOf(options.type().toUpperCase()));
        if (options.configJson() != null) {
            request.config(JSON.readValue(options.configJson(), new TypeReference<Map<String, Object>>() {}));
        } else if (options.configFile() != null) {
            request.config(JSON.readValue(Files.readString(options.configFile()), new TypeReference<Map<String, Object>>() {}));
        }
        Response<ServeResponse> response = api.serveModel(request).execute();
        requireSuccess(response, "serve");
        ServeResponse body = response.body();
        System.out.println("Serving " + body.getOwner() + "/" + body.getName() + " (" + body.getType() + ")");
    }

    private static void requireSuccess(Response<?> response, String operation) throws IOException {
        if (response.isSuccessful() && response.body() != null) {
            return;
        }
        String detail = response.errorBody() == null ? response.message() : response.errorBody().string();
        throw new IOException(operation + " failed (HTTP " + response.code() + "): " + detail);
    }

    private static ModelsApi modelsApi(String endpoint) {
        ApiClient apiClient = new ApiClient();
        apiClient.setAdapterBuilder(new Retrofit.Builder()
                .baseUrl(endpoint.endsWith("/") ? endpoint : endpoint + "/")
                .client(new OkHttpClient.Builder().build())
                .addConverterFactory(ScalarsConverterFactory.create())
                .addConverterFactory(JacksonConverterFactory.create()));
        return apiClient.createService(ModelsApi.class);
    }

    private static void usage() {
        System.out.println("Usage:");
        System.out.println("  inferj-cli fetch OWNER MODEL [--endpoint URL]");
        System.out.println("  inferj-cli serve OWNER MODEL --type TYPE [--config-json JSON] [--config FILE]");
    }

    private record Options(String endpoint, String owner, String name, String type,
                           String configJson, Path configFile) {
        static Options parse(String[] args) {
            if (args.length < 3) {
                throw new IllegalArgumentException("owner and model are required");
            }
            String endpoint = "http://localhost:8080";
            String type = null;
            String configJson = null;
            Path configFile = null;
            Map<String, String> flags = new HashMap<>();
            for (int i = 3; i < args.length; i++) {
                String arg = args[i];
                if (!arg.startsWith("--") || i + 1 >= args.length) {
                    throw new IllegalArgumentException("Invalid option: " + arg);
                }
                flags.put(arg, args[++i]);
            }
            endpoint = flags.getOrDefault("--endpoint", endpoint);
            type = flags.get("--type");
            configJson = flags.get("--config-json");
            String configPath = flags.get("--config");
            if (configPath != null) {
                configFile = Path.of(configPath);
            }
            if ("serve".equals(args[0]) && type == null) {
                throw new IllegalArgumentException("--type is required for serve");
            }
            if (configJson != null && configFile != null) {
                throw new IllegalArgumentException("use either --config-json or --config");
            }
            return new Options(endpoint, args[1], args[2], type, configJson, configFile);
        }
    }
}
