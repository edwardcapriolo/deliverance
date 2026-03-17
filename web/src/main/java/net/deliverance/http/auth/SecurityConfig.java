package net.deliverance.http.auth;


import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;

import java.util.Optional;

@Configuration
@EnableWebSecurity
public class SecurityConfig {

    private String user;
    Optional<JwtAuthenticationFilter> jwtAuthenticationFilterOptional;

    public SecurityConfig(@Value("${deliverance.basic.auth.user:#{null}}") String user,
                          Optional<JwtAuthenticationFilter> jwtAuthenticationFilterOptional){
        this.user = user;
        this.jwtAuthenticationFilterOptional = jwtAuthenticationFilterOptional;
    }

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        if (user == null &&  jwtAuthenticationFilterOptional.isEmpty()) {
            return http.build();
        }
        http
                .securityMatcher("/chat/**", "/embeddings/**")
                .authorizeHttpRequests(auth -> auth
                        .anyRequest().authenticated())
                .csrf(AbstractHttpConfigurer::disable)
                .sessionManagement(session -> session
                        .sessionCreationPolicy(SessionCreationPolicy.STATELESS)
                );
        if (user != null){
              http.httpBasic(basic -> basic
                    .realmName("deliverance")
                    .authenticationEntryPoint((request, response, authException) -> {
                        response.setHeader("WWW-Authenticate", "Basic realm=\"deliverance\"");
                        response.setStatus(401);
                        response.getWriter().write("{\"error\": \"Unauthorized\"}");
                    })
            );
        }
        jwtAuthenticationFilterOptional.ifPresent(jwtAuthenticationFilter -> {
            http.addFilterBefore(jwtAuthenticationFilter, UsernamePasswordAuthenticationFilter.class);
        });
        return http.build();
    }
}