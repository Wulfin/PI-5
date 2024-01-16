// NavBar.js

import React from 'react';
import { Link } from 'react-router-dom';
import './NavBar.css';

const NavBar = () => {
    return (
        <nav className='navbar'>
            <div className='navbar-logo'>
                <img className='twitter-logo' src='twitter-logo.png' alt='Twitter Logo' />
                <span>Twitter Clone</span>
            </div>
            <ul>
                <li><Link to="/">Home</Link></li>
                <li><Link to="/tweets">Tweets</Link></li>
                <li><Link to="/request-settings">Request Settings</Link></li>
            </ul>
        </nav>
    );
};

export default NavBar;
