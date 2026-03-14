package net.deliverance.http;


import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

@ConditionalOnProperty("deliverance.basic.auth.user")
@Service
public class CustomUserDetailsService implements UserDetailsService {

    private final String user;
    private final String pass;
    public CustomUserDetailsService(@Value("${deliverance.basic.auth.user}") String user,
                                    @Value("${deliverance.basic.auth.pass}") String pass) {
        this.user = user;
        this.pass = pass;
    }

    // Spring Security calls this method during authentication
    @Override
    public UserDetails loadUserByUsername(String username)
            throws UsernameNotFoundException {
        if (!username.equals(user)) {
            throw new UsernameNotFoundException("no such user username");
        }
        return User.builder()
                .username(user)
                .password(passwordEncoder().encode(pass))
                .roles("user")
                .accountExpired(false)
                .accountLocked(false)
                .credentialsExpired(false)
                .disabled(false)
                .build();
    }

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder(12);
    }
}