import React, {useEffect, useState} from 'react';
import axios from 'axios';
import './FetchData.css';

const FetchData = () => {
    const [data, setData] = useState([]);
    const [searchTerm, setSearchTerm] = useState('');
    const [selectedSentiment, setSelectedSentiment] = useState('');
    const [startDate, setStartDate] = useState('');
    const [endDate, setEndDate] = useState('');

    const isWithinRange = (date, start, end) => {
        if (!start && !end) return true;
        if (start && !end) return date >= start;
        if (!start && end) return date <= end;

        return date >= start && date <= end;
    };

    const fetchData = async () => {
        try {
            const response = await axios.get('http://localhost:8081/tweets');
            setData(response.data);
        } catch (error) {
            console.error('Error fetching data:', error);
        }
    };

    const handleRefresh = () => {
        fetchData();
    };

    const handleClearDatabase = async () => {
        try {
            await axios.delete('http://localhost:8081/tweets/delete/all');
            fetchData();
        } catch (error) {
            console.error('Error clearing database:', error);
        }
    };

    useEffect(() => {
        fetchData();
    }, []);

    const handleSearchChange = (event) => {
        setSearchTerm(event.target.value);
    };

    const handleSentimentChange = (event) => {
        setSelectedSentiment(event.target.value);
    };

    const handleStartDateChange = (event) => {
        setStartDate(event.target.value);
    };

    const handleEndDateChange = (event) => {
        setEndDate(event.target.value);
    };

    const filteredData = data.filter((item) => {
        const matchesSearch = item.username.toLowerCase().includes(searchTerm.toLowerCase());
        const matchesSentiment =
            selectedSentiment === '' || item.sentiment.toLowerCase() === selectedSentiment.toLowerCase();
        const isWithinDateRange = isWithinRange(item.timestamp, startDate, endDate);

        return matchesSearch && matchesSentiment && isWithinDateRange;
    });

    return (
        <div className='page-container'>
            <div className='content-container'>
                <div className='search-filter-container'>
                    <h4>Search by Username:</h4>
                    <input type='text' placeholder='Enter username' value={searchTerm} onChange={handleSearchChange}/>
                </div>
                <div className='search-filter-container'>
                    <h4>Filter by Sentiment:</h4>
                    <select value={selectedSentiment} onChange={handleSentimentChange}>
                        <option value=''>All Sentiments</option>
                        <option value='positive'>Positive</option>
                        <option value='negative'>Negative</option>
                    </select>
                </div>
                <div className='search-filter-container'>
                    <h4>Filter by Date Range:</h4>
                    <div>
                        <label>Start Date:</label>
                        <input type='date' value={startDate} onChange={handleStartDateChange}/>
                    </div>
                    <div>
                        <label>End Date:</label>
                        <input type='date' value={endDate} onChange={handleEndDateChange}/>
                    </div>
                </div>
                <div className='button-container'>
                    <button onClick={handleRefresh}>Refresh Page</button>
                    <button onClick={handleClearDatabase}>Clear Database</button>
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
                    {filteredData.map(({id, timestamp, username, content, sentiment}) => (
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
