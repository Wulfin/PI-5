import React from 'react';
import './App.css';
import FetchData from './FetchData';
import NavBar from './NavBar';
import Home from "./Home";
import {BrowserRouter as Router, Route, Switch} from "react-router-dom";

function App() {
    return (
        <Router>
            <div className="App">
                <NavBar/>
                <div className="content">
                    <Switch>
                        <Route exact path="/">
                            <Home/>
                        </Route>
                        <Route exact path="/tweets">
                            <FetchData/>
                        </Route>
                        <Route exact path="/request-settings">
                            <FetchData/>
                        </Route>
                    </Switch>
                </div>
            </div>
        </Router>
    );
}

export default App;
