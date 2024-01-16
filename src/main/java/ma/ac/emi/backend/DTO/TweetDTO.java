package ma.ac.emi.backend.DTO;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;

import java.time.LocalDateTime;

@Getter
@Setter
@ToString
@RequiredArgsConstructor
@Builder
@AllArgsConstructor
@JsonIgnoreProperties(ignoreUnknown = true)
public class TweetDTO {

    @JsonProperty("datetime")
    private LocalDateTime timestamp;
    @JsonProperty("username")
    private String username;
    @JsonProperty("content")
    private String content;
    @JsonProperty("sentiment")
    private String sentiment;
    @JsonProperty("retweets")
    private String retweets;
    @JsonProperty("likes")
    private String likes;

}
