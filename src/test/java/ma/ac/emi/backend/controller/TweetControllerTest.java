package ma.ac.emi.backend.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import ma.ac.emi.backend.entity.Tweet;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders;
import org.springframework.test.web.servlet.result.MockMvcResultMatchers;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;
import java.util.List;

import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@WebMvcTest(TweetController.class)
public class TweetControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Test
    public void testSaveTweet() throws Exception {
        // Using the builder pattern
        Tweet tweet = Tweet.builder()
                .username("testUser")
                .content("This is a test tweet content")
                .sentiment("Positive")
                .timestamp(LocalDateTime.parse("2023-12-05 10:30:00", DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")))
                .build();

        mockMvc.perform(MockMvcRequestBuilders.post("/tweets")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(tweet)))
                .andExpect(status().isCreated());
    }

    @Test
    public void testSaveTweets() throws Exception {
        // Using the builder pattern
        Tweet tweet1 = Tweet.builder()
                .username("testUser1")
                .content("This is a test tweet content 1")
                .sentiment("Positive")
                .timestamp(LocalDateTime.now())
                .build();

        Tweet tweet2 = Tweet.builder()
                .username("testUser2")
                .content("This is a test tweet content 2")
                .sentiment("Negative")
                .timestamp(LocalDateTime.now())
                .build();

        List<Tweet> tweets = Arrays.asList(tweet1, tweet2);

        mockMvc.perform(MockMvcRequestBuilders.post("/tweets/batch")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(tweets)))
                .andExpect(status().isCreated());
    }

    @Test
    public void testFindTweetsByUsername() throws Exception {
        mockMvc.perform(MockMvcRequestBuilders.get("/tweets/username/testUser"))
                .andExpect(status().isOk())
                .andExpect(MockMvcResultMatchers.jsonPath("$.length()").value(1));
    }

    @Test
    public void testFindTweetById() throws Exception {
        mockMvc.perform(MockMvcRequestBuilders.get("/tweets/1"))
                .andExpect(status().isOk());
    }

    @Test
    public void testFindAllTweets() throws Exception {
        mockMvc.perform(MockMvcRequestBuilders.get("/tweets"))
                .andExpect(status().isOk())
                .andExpect(MockMvcResultMatchers.jsonPath("$.length()").value(1));
    }

    @Test
    public void testFindTweetsByDate() throws Exception {
        mockMvc.perform(MockMvcRequestBuilders.get("/tweets/byDate")
                        .param("startDate", "2023-12-01 00:00:00")
                        .param("endDate", "2023-12-31 23:59:59"))
                .andExpect(status().isOk())
                .andExpect(MockMvcResultMatchers.jsonPath("$.length()").value(1));
    }

    @Test
    public void testDeleteTweetById() throws Exception {
        mockMvc.perform(MockMvcRequestBuilders.delete("/tweets/1"))
                .andExpect(status().isOk());
    }
}
