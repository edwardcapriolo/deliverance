package net.deliverance.http;


import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.web.SecurityFilterChain;

@Configuration
@EnableWebSecurity
public class SecurityConfig {

    private String user;

    public SecurityConfig(@Value("${deliverance.basic.auth.user:#{null}}") String user){
        this.user = user;
    }
    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        if (user==null){
            return http.build();
        }
        http
                .securityMatcher("/chat/**")
                .authorizeHttpRequests(auth -> auth
                        .anyRequest().authenticated())
                .httpBasic(basic -> basic
                        .realmName("deliverance")
                        .authenticationEntryPoint((request, response, authException) -> {
                            response.setHeader("WWW-Authenticate", "Basic realm=\"deliverance\"");
                            response.setStatus(401);
                            response.getWriter().write("{\"error\": \"Unauthorized\"}");
                        })
                )
                .csrf(AbstractHttpConfigurer::disable)
                .sessionManagement(session -> session
                        .sessionCreationPolicy(SessionCreationPolicy.STATELESS)
                );

        return http.build();
    }
}