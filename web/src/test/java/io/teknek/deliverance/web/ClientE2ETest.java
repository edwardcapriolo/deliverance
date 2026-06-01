package io.teknek.deliverance.web;

import io.teknek.deliverance.client.api.ModelsApi;
import io.teknek.deliverance.client.model.ListModelsResponse;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import retrofit2.Retrofit;
import retrofit2.converter.jackson.JacksonConverterFactory;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;

import static org.assertj.core.api.Assertions.assertThat;

@SpringBootTest(classes = net.deliverance.http.DeliveranceApplication.class,
        webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
public class ClientE2ETest {

    @LocalServerPort
    int port;

    @Test
    void shouldListModelsViaClient() throws Exception {
        String baseUrl = "http://localhost:" + port + "/";

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

        ListModelsResponse response = api.listModels().execute().body();

        assertThat(response).isNotNull();
        assertThat(response.getData()).isNotNull();
    }
}
