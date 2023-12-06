package ma.ac.emi.backend.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import ma.ac.emi.backend.DTO.TweetDTO;
import ma.ac.emi.backend.repository.TweetRepository;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

@Service
public class TweetService {

    private final TweetRepository tweetRepository;
    private final WebClient webClient;

    public TweetService(TweetRepository tweetRepository,
                        @Value("${url}") String apiUrl) {
        this.tweetRepository = tweetRepository;
        this.webClient = WebClient.create(apiUrl);
    }


}
