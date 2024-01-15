import React, {useEffect, useState} from 'react';
import axios from 'axios';
import './FetchData.css'; // Import your custom CSS file for styling

const FetchData = () => {
    const [data, setData] = useState([]);

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

    return (
        <div className='page-container'>
            <div className='content-container'>
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
                    {data.map(({id, timestamp, username, content, sentiment}) => (
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
