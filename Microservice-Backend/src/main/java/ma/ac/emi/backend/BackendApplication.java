package ma.ac.emi.backend;

import ma.ac.emi.backend.entity.Tweet;
import ma.ac.emi.backend.repository.TweetRepository;
import ma.ac.emi.backend.service.TweetService;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.openfeign.EnableFeignClients;
import org.springframework.context.annotation.Bean;
import org.springframework.web.reactive.function.client.WebClient;

import java.util.List;

@SpringBootApplication
@EnableFeignClients
public class BackendApplication {

    public static void main(String[] args) {
        SpringApplication.run(BackendApplication.class, args);
    }



}
