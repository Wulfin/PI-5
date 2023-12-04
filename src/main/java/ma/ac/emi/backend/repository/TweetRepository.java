package ma.ac.emi.backend.repository;

import ma.ac.emi.backend.entity.Tweet;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.rest.core.annotation.RepositoryRestResource;
import org.springframework.data.rest.webmvc.RepositoryRestController;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@RepositoryRestResource
public interface TweetRepository extends JpaRepository<Tweet, Long> {

    List<Tweet> findByUsername(String username);

    List<Tweet> findByTimestampBetween(LocalDateTime startDate, LocalDateTime endDate);

    /*List<Tweet> saveAllTweets(Iterable<Tweet> tweets);*/
}