package net.deliverance.http.auth;



import io.jsonwebtoken.Claims;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.io.Decoders;
import io.jsonwebtoken.security.Keys;

import java.nio.charset.StandardCharsets;
import java.security.Key;
import java.util.Date;
import java.util.function.Function;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Service;

import javax.crypto.SecretKey;

@ConditionalOnProperty("security.jwt.secret-key")
@Service
public class JwtService {
    @Value("${security.jwt.secret-key}")
    private String secretKey;

    //public String extractUsername(String token) {
    //    return extractClaim(token, Claims::getSubject);
    //}

    public <T> T extractClaim(String token, Function<Claims, T> claimsResolver) {
        final Claims claims = extractAllClaims(token);
        return claimsResolver.apply(claims);
    }


    private boolean isTokenExpired(String token) {
        return extractExpiration(token).before(new Date());
    }

    private Date extractExpiration(String token) {
        return extractClaim(token, Claims::getExpiration);
    }


    public Claims extractAllClaims(String token) {
        //Jwt<?,?> parsed = Jwts.parser().build().parse(token);
        return Jwts.parser().setSigningKey(getSigningKey()).build().parseSignedClaims(token).getPayload();

    }

    private Key getSignInKey() {
        byte[] keyBytes = Decoders.BASE64.decode(secretKey);
        return Keys.hmacShaKeyFor(keyBytes);
    }

    private static SecretKey getSigningKey() {
        // JJWT v0.13 recommends using the Keys class for generating keys
        byte[] keyBytes ="".getBytes(StandardCharsets.UTF_8);
        return Keys.hmacShaKeyFor(keyBytes);
    }
}