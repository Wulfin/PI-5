package ma.ac.emi.backend.controller;

import ma.ac.emi.backend.entity.Tweet;
import ma.ac.emi.backend.repository.TweetRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.List;

@RestController
@RequestMapping("/tweets")
public class TweetController {

    private final TweetRepository tweetRepository;

    @Autowired
    public TweetController(TweetRepository tweetRepository) {
        this.tweetRepository = tweetRepository;
    }

    @GetMapping
    public ResponseEntity<List<Tweet>> findAllTweets() {
        try {
            List<Tweet> tweets = tweetRepository.findAll();
            return new ResponseEntity<>(tweets, HttpStatus.OK);
        } catch (Exception e) {
            return new ResponseEntity<>(HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @GetMapping("/username/{username}")
    public ResponseEntity<List<Tweet>> findTweetsByUsername(@PathVariable("username") String username) {
        try {
            List<Tweet> tweets = tweetRepository.findByUsername(username);
            return new ResponseEntity<>(tweets, HttpStatus.OK);
        } catch (Exception e) {
            return new ResponseEntity<>(HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @GetMapping("/{id}")
    public ResponseEntity<String> findTweetbyID(@PathVariable("id") Long id) {
        try {
            // findById returns an Optional<Tweet>
            java.util.Optional<Tweet> tweetOptional = tweetRepository.findById(id);

            // Check if tweet exists
            if (tweetOptional.isPresent()) {
                Tweet tweet = tweetOptional.get();
                // Do something with the tweet if needed
                return new ResponseEntity<>("Tweet found: " + tweet.getContent(), HttpStatus.OK);
            } else {
                // tweet with ID is not found
                return new ResponseEntity<>("Tweet not found", HttpStatus.NOT_FOUND);
            }
        } catch (Exception e) {
            // Handle exceptions if any
            return new ResponseEntity<>("Error occurred: " + e.getMessage(), HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @GetMapping("/byDate")
    public ResponseEntity<List<Tweet>> getTweetsByDate(
            @RequestParam("startDate") @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startDate,
            @RequestParam("endDate") @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endDate) {
        try {
            List<Tweet> tweets = tweetRepository.findByTimestampBetween(startDate, endDate);
            return new ResponseEntity<>(tweets, HttpStatus.OK);
        } catch (Exception e) {
            return new ResponseEntity<>(HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }


    @DeleteMapping("/{id}")
    public ResponseEntity<String> deleteTweetById(@PathVariable("id") Long id) {
        try {
            tweetRepository.deleteById(id);
            return new ResponseEntity<>("Tweet deleted successfully", HttpStatus.OK);
        } catch (Exception e) {
            return new ResponseEntity<>("Error deleting tweet: " + e.getMessage(), HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @PostMapping()
    public ResponseEntity<String> saveTweet(@RequestBody Tweet tweet) {
        try {
            // If the timestamp is not provided in the JSON, set it to the current date and time
            if (tweet.getTimestamp() == null) {
                tweet.setTimestamp(LocalDateTime.now());
            }

            // Save the tweet using the repository
            tweetRepository.save(tweet);

            return new ResponseEntity<>("Tweet saved successfully", HttpStatus.CREATED);
        } catch (Exception e) {
            return new ResponseEntity<>("Error saving tweet: " + e.getMessage(), HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @PostMapping("/batch")
    public ResponseEntity<String> saveTweets(@RequestBody List<Tweet> tweets) {
        try {
            // Save each tweet in the list
            tweetRepository.saveAll(tweets);

            return new ResponseEntity<>("Tweets saved successfully", HttpStatus.CREATED);
        } catch (Exception e) {
            return new ResponseEntity<>("Error saving tweets: " + e.getMessage(), HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }


}
