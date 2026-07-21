package io.teknek.deliverance.web;

import java.time.Duration;

import io.teknek.deliverance.client.api.ModelsApi;
import io.teknek.deliverance.client.model.ListModelsResponse;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import retrofit2.Retrofit;
import retrofit2.converter.jackson.JacksonConverterFactory;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;

import static org.assertj.core.api.Assertions.assertThat;

@SpringBootTest(classes = net.deliverance.http.DeliveranceApplication.class,
        webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
public class ClientE2ETest {

    @LocalServerPort
    int port;

    @ParameterizedTest
    @ValueSource(strings = { "retrofit", "spring" })
    void shouldListModelsViaClient(String client) throws Exception {
        String baseUrl = "http://localhost:" + port + "/";

        if ("retrofit".equals(client)) {
            ListModelsResponse response = listModelsWithRetrofitClient(baseUrl);

            assertThat(response).isNotNull();
            assertThat(response.getData()).isNotNull();
        }
        else {
            io.teknek.deliverance.client.spring.model.ListModelsResponse response = listModelsWithSpringClient(baseUrl);

            assertThat(response).isNotNull();
            assertThat(response.getData()).isNotNull();
        }
    }

    private ListModelsResponse listModelsWithRetrofitClient(String baseUrl) throws Exception {

        String auth = java.util.Base64.getEncoder().encodeToString("1:2".getBytes());

        OkHttpClient httpClient = new OkHttpClient.Builder()
                .addInterceptor(chain -> {
                    Request request = chain.request().newBuilder()
                            .addHeader("Authorization", "Basic " + auth)
                            .addHeader("Content-Type", "application/json")
                            .addHeader("Accept", "application/json")
                            .build();
                    return chain.proceed(request);
                })
                .build();

        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl(baseUrl)
                .client(httpClient)
                .addConverterFactory(JacksonConverterFactory.create())
                .build();

        ModelsApi api = retrofit.create(ModelsApi.class);

        return api.listModels().execute().body();
    }

    private io.teknek.deliverance.client.spring.model.ListModelsResponse listModelsWithSpringClient(String baseUrl) {
        String auth = java.util.Base64.getEncoder().encodeToString("1:2".getBytes());
        io.teknek.deliverance.client.spring.core.ApiClient apiClient = new io.teknek.deliverance.client.spring.core.ApiClient()
                .setBasePath(baseUrl)
                .addDefaultHeader("Authorization", "Basic " + auth)
                .addDefaultHeader("Content-Type", "application/json")
                .addDefaultHeader("Accept", "application/json");
        io.teknek.deliverance.client.spring.api.ModelsApi api = new io.teknek.deliverance.client.spring.api.ModelsApi(
                apiClient);

        return api.listModels().block(Duration.ofSeconds(5));
    }
}
