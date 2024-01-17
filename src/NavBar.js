// NavBar.js

import React from 'react';
import { Link } from 'react-router-dom';
import './NavBar.css';

const NavBar = () => {
    return (
        <nav className='navbar'>
            <div className="navbar-title">
                <Link to="/" className="title-link">
                    Sentiment Analysis
                </Link>
            </div>
            <ul className="navbar-links">
                <li><Link to="/" className="nav-link">Home</Link></li>
                <li><Link to="/tweets" className="nav-link">Tweets</Link></li>
                <li><Link to="/graphics " className="nav-link">Graphics</Link></li>
                <li><Link to="/request-settings" className="nav-link">Request Settings</Link></li>
            </ul>
        </nav>
    );
};

export default NavBar;
