import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './FetchData.css';

const FetchData = () => {
    const [data, setData] = useState([]);
    const [searchTerm, setSearchTerm] = useState('');
    const [selectedSentiment, setSelectedSentiment] = useState('');

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await axios.get('http://localhost:8081/tweets');
                setData(response.data);
            } catch (error) {
                console.error('Error fetching data:', error);
            }
        };

        fetchData();
    }, []);

    const handleSearchChange = (event) => {
        setSearchTerm(event.target.value);
    };

    const handleSentimentChange = (event) => {
        setSelectedSentiment(event.target.value);
    };

    const filteredData = data.filter((item) => {
        const matchesSearch = item.username.toLowerCase().includes(searchTerm.toLowerCase());
        const matchesSentiment =
            selectedSentiment === '' || item.sentiment.toLowerCase() === selectedSentiment.toLowerCase();

        return matchesSearch && matchesSentiment;
    });

    return (
        <div className='page-container'>
            <div className='content-container'>
                <div className='search-filter-container'>
                    <h4>Search by Username:</h4>
                    <input
                        type='text'
                        placeholder='Enter username'
                        value={searchTerm}
                        onChange={handleSearchChange}
                    />
                </div>
                <div className='search-filter-container'>
                    <h4>Filter by Sentiment:</h4>
                    <select value={selectedSentiment} onChange={handleSentimentChange}>
                        <option value=''>All Sentiments</option>
                        <option value='positive'>Positive</option>
                        <option value='negative'>Negative</option>
                    </select>
                </div>
                <h3>Tweets Sauvegardés</h3>
                <table className='styled-table'>
                    <thead>
                    <tr>
                        <th>ID</th>
                        <th>Time Stamp</th>
                        <th>Username</th>
                        <th>Content</th>
                        <th>Sentiment</th>
                    </tr>
                    </thead>
                    <tbody>
                    {filteredData.map(({ id, timestamp, username, content, sentiment }) => (
                        <tr key={id}>
                            <td>{id}</td>
                            <td>{timestamp}</td>
                            <td>{username}</td>
                            <td>{content}</td>
                            <td>{sentiment}</td>
                        </tr>
                    ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default FetchData;
