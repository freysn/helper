#ifndef __HELPER_SIGNAL__
#define __HELPER_SIGNAL__

#include <signal.h>
#include <errno.h>
#include <stdexcept>

namespace helper
{
using std::runtime_error;

class SignalException : public runtime_error
{
public:
   SignalException(const std::string& _message)
      : std::runtime_error(_message)
   {}
};

class SignalHandler
{
protected:
    static bool mbGotExitSignal;

public:
    SignalHandler();
    ~SignalHandler();

    static bool gotExitSignal();
    static void setExitSignal(bool _bExitSignal);

    void        setupSignalHandlers();
    static void exitSignalHandler(int _ignored);

};

          
bool SignalHandler::mbGotExitSignal = false;

/**
* Default Contructor.
*/
SignalHandler::SignalHandler()
{
  setupSignalHandlers();
}

/**
* Destructor.
*/
SignalHandler::~SignalHandler()
{
}

/**
* Returns the bool flag indicating whether we received an exit signal
* @return Flag indicating shutdown of program
*/
bool SignalHandler::gotExitSignal()
{
    return mbGotExitSignal;
}

/**
* Sets the bool flag indicating whether we received an exit signal
*/
void SignalHandler::setExitSignal(bool _bExitSignal)
{
    mbGotExitSignal = _bExitSignal;
}

/**
* Sets exit signal to true.
* @param[in] _ignored Not used but required by function prototype
*                     to match required handler.
*/
void SignalHandler::exitSignalHandler(int _ignored)
{
    mbGotExitSignal = true;
}

/**
* Set up the signal handlers for CTRL-C.
*/
void SignalHandler::setupSignalHandlers()
{
    if (signal((int) SIGINT, SignalHandler::exitSignalHandler) == SIG_ERR)
    {
        throw SignalException("!!!!! Error setting up signal handlers !!!!!");
    }
}
}

#endif //__HELPER_SIGNAL__
