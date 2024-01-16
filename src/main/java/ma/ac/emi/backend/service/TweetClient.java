package ma.ac.emi.backend.service;

import ma.ac.emi.backend.DTO.TweetDTO;
import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;

import java.util.List;

@FeignClient(name = "TweetClient", url = "http://localhost:5000")
public interface TweetClient {

    @GetMapping("/sentiments/{searchQuery}")
    List<TweetDTO> getTweets(@PathVariable("searchQuery")String q);
}
