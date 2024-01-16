import React, { useState } from 'react';
import axios from 'axios';
import './Request.css';

const Request = () => {
    const [searchQuery, setSearchQuery] = useState('');
    const [loading, setLoading] = useState(false);
    const [successMessage, setSuccessMessage] = useState('');
    const [errorMessage, setErrorMessage] = useState('');

    const handleSubmit = async (event) => {
        event.preventDefault();
        setLoading(true);

        try {
            const response = await axios.get(`http://localhost:8081/tweets/fetch/${encodeURIComponent(searchQuery)}`);
            setSuccessMessage(response.data.message); // Assuming the server sends a message upon success
        } catch (error) {
            setErrorMessage('An error occurred while fetching tweets.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="twitter-search-form-container">
            <h2>Twitter Search Form</h2>
            <form onSubmit={handleSubmit}>
                <div className="search-input-container">
                    <label>Search Query:</label>
                    <input
                        type="text"
                        placeholder="Enter your search query"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                    />
                </div>
                <button type="submit" className="search-button">
                    {loading ? (
                        <div className="loading-spinner"></div>
                    ) : (
                        'Search on Twitter'
                    )}
                </button>

                {loading && <div className="loading-message">Loading...</div>}
                {successMessage && <div className="success-message">{successMessage}</div>}
                {errorMessage && <div className="error-message">{errorMessage}</div>}
            </form>
        </div>
    );
};

export default Request;