package ma.ac.emi.backend.DTO;

import lombok.*;

import java.time.LocalDateTime;

@Getter
@Setter
@ToString
@RequiredArgsConstructor
@Builder
@AllArgsConstructor
public class TweetDTO {

    private Long id;

    private LocalDateTime timestamp;

    private String username;

    private String content;

    private String sentiment;

}
