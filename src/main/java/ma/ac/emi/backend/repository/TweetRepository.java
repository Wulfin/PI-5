package ma.ac.emi.backend.repository;

import ma.ac.emi.backend.entity.Tweet;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.rest.core.annotation.RepositoryRestResource;
import org.springframework.data.rest.webmvc.RepositoryRestController;

@RepositoryRestResource
public interface TweetRepository extends JpaRepository<Tweet, Long> {
}