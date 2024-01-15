// NavBar.js

import React from 'react';
import {Link} from 'react-router-dom';
import './NavBar.css';

const NavBar = () => {
    return (
        <nav className='navbar'>
            <ul>
                <li><Link to="/">Home</Link></li>
                <li><Link to="/tweets">Tweets</Link></li>
                <li><Link to="/request-settings">Request Settings</Link></li>
            </ul>
        </nav>
    );
};

export default NavBar;
