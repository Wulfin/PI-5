package ma.ac.emi.backend;

import ma.ac.emi.backend.entity.Tweet;
import ma.ac.emi.backend.repository.TweetRepository;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

@SpringBootApplication
public class BackendApplication {

    public static void main(String[] args) {
        SpringApplication.run(BackendApplication.class, args);
    }

    @Bean
    CommandLineRunner commandLineRunner(){

        return args -> {
            /*Tweet tweet1 = Tweet.builder()
                    .content("Contenu tweet")
                    .sentiment("positif")
                    .username("souhailxx")
                    .build();
            tweetRepository.save(tweet1);
            Tweet tweet2 = Tweet.builder()
                    .content("Contenu tweet 2")
                    .sentiment("negatif")
                    .username("dearze")
                    .build();
            tweetRepository.save(tweet2);*/

        };
    }

}
