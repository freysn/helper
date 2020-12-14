#ifndef __HELPER_PRAGMA_MESSAGES__
#define __HELPER_PRAGMA_MESSAGES__

#define Stringize( L )     #L 
#define MakeString( M, L ) M(L)
#define $Line MakeString( Stringize, __LINE__ )
#define TODO "TODO: "
#define TODOL __FILE__ "(" $Line ") ; TODO: "

//#pragma message(TODO "something")

#endif //__HELPER_PRAGMA_MESSAGES__
