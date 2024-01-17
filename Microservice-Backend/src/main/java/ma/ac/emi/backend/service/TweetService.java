package ma.ac.emi.backend.service;

import ma.ac.emi.backend.DTO.TweetDTO;
import ma.ac.emi.backend.entity.Tweet;
import ma.ac.emi.backend.repository.TweetRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
public class TweetService {

    private final TweetRepository tweetRepository;
    private final WebClient webClient;

    @Autowired
    private TweetClient tweetClient;

    @Value("${url}")
    private String apiUrl;

    public TweetService(TweetRepository tweetRepository, WebClient.Builder webClientBuilder) {
        this.tweetRepository = tweetRepository;
        this.webClient = webClientBuilder.baseUrl(apiUrl).build();
    }

    public Tweet mapToEntity(TweetDTO tweetDTO) {
        return Tweet.builder()
                .timestamp(tweetDTO.getTimestamp())
                .username(tweetDTO.getUsername())
                .content(tweetDTO.getContent())
                .sentiment(tweetDTO.getSentiment())
                .build();
    }
    public List<Tweet> fetchTweets(String q) {
        List<TweetDTO> tweetsDTO = tweetClient.getTweets(q);

        List<Tweet> tweets = tweetsDTO.stream()
                .map(this::mapToEntity)
                .collect(Collectors.toList());

        return tweets;
    }


    public Tweet saveTweet(Tweet tweet) {
        tweet.setContent(Optional.ofNullable(tweet.getContent()).orElse("empty"));
        tweet.setSentiment(Optional.ofNullable(tweet.getSentiment()).orElse("Positive"));
        tweet.setUsername(Optional.ofNullable(tweet.getUsername()).orElse("empty"));
        tweet.setTimestamp(Optional.ofNullable(tweet.getTimestamp()).orElse(LocalDateTime.parse("2023-12-05T10:30:30")));
        return tweetRepository.save(tweet);
    }




}
